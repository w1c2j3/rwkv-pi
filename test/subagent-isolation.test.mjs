// Real sandbox and subprocesses, mock model only: this tests isolation, not model quality.
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { SubagentManager } from "../dist/subagents.js";

test("isolated runtime paths reject aliases for the host root", async t => {
  const root = await mkdtemp(join(tmpdir(), "rwkv-runtime-path-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const alias = join(root, "root-link");
  await symlink("/", alias);
  for (const path of ["/", "/.", "/tmp/..", alias, "relative", join(root, "missing")]) {
    assert.throws(() => new SubagentManager(root, {
      concurrency: 1, maxRequests: 1, timeoutSeconds: 1, maxTasks: 1,
      sandbox: "bwrap", mount: "/srv/test-project", runtimePaths: [path],
    }, join(root, "runs")), /Invalid isolated runtime path/);
  }
});

test("an isolated child sees its own project at an unseen path, not sibling projects", {timeout:120000}, async t => {
  const root=await mkdtemp(join(tmpdir(),"rwkv-child-isolation-"));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const cwd=join(root,"project");await mkdir(cwd);
  const probe=spawnSync("bwrap",["--unshare-pid","--ro-bind","/","/","--","/bin/true"],{encoding:"utf8"});
  if(probe.status!==0){
    const reason=String(probe.error??probe.stderr);
    if(process.env.REQUIRE_SANDBOX_TESTS==="1"||!/ENOENT|Operation not permitted|No permissions to create/.test(reason))throw Error(reason);
    t.skip(reason);return;
  }
  const outside=join(root,"sibling.txt");
  await writeFile(outside,"outside-marker\n");await writeFile(join(cwd,"input.txt"),"INSIDE\n");
  const server=createServer(async(req,res)=>{
    let raw="";for await(const c of req)raw+=c;
    if(req.url.endsWith("/tokens/count")){res.end('{"tokens":100}');return;}
    if(req.url.endsWith("/server/stop")){res.end('{}');return;}
    const prompt=JSON.parse(raw).contents[0];
    const output=prompt.includes("Successfully wrote")?"完成。":prompt.includes("INSIDE")
      ?'<tool_call>'+JSON.stringify({name:"write",arguments:{path:"out.txt",content:"OK\n"}})+'</tool_call>'
      :prompt.includes("ENOENT")?'<tool_call>'+JSON.stringify({name:"read",arguments:{path:"input.txt"}})+'</tool_call>'
      :'<tool_call>'+JSON.stringify({name:"read",arguments:{path:outside}})+'</tool_call>';
    res.writeHead(200,{"content-type":"text/event-stream"});
    const emit=(delta,finish_reason=null)=>res.write(`data: ${JSON.stringify({id:"fixture",object:"chat.completion.chunk",choices:[{index:0,delta,finish_reason,...(finish_reason?{matched_stop:"<EOD>"}:{})}]})}\n\n`);
    emit({content:"</think>\n"+output+"<EOD>"});emit({},"stop");res.end("data: [DONE]\n\n");
  });
  server.listen(0,"127.0.0.1");await once(server,"listening");
  t.after(()=>new Promise(r=>{server.close(r);server.closeAllConnections();}));
  const runtime=fileURLToPath(new URL("../",import.meta.url));
  const repo=resolve(process.env.RWKV_PI_SOURCE??".upstream/pi","../..");
  Object.assign(process.env,{RWKV_BASE_URL:`http://127.0.0.1:${server.address().port}/v1`,RWKV_MODEL:"mock",
    RWKV_CONTEXT_WINDOW:"32768",RWKV_MAX_TOKENS:"4096",RWKV_TOOL_TOLERANCE:"0",PI_OFFLINE:"1"});
  const mount=`/srv/child-${randomUUID()}/project`;
  const manager=new SubagentManager(cwd,{concurrency:1,maxRequests:6,timeoutSeconds:90,maxTasks:1,sandbox:"bwrap",mount,translateTaskPaths:false,
    runtimePaths:[join(runtime,"dist"),join(runtime,"node_modules"),join(runtime,"package.json"),join(repo,"node_modules"),join(repo,".upstream/pi"),dirname(dirname(process.execPath))]},join(root,"runs"));
  t.after(()=>manager.shutdown());
  manager.start([{task:`读取 input.txt，然后创建 out.txt。原始指令中的路径字符串必须保持原样：${cwd}/literal。`,write:true}]);
  const [worker]=await manager.wait(undefined,100000);
  assert.equal(worker.state,"completed",worker.error);
  const events=(await readFile(join(worker.dir,"events.jsonl"),"utf8")).split('\n').filter(Boolean).map(JSON.parse);
  const results=events.filter(e=>e.type==="message_end"&&e.message?.role==="toolResult").map(e=>e.message);
  assert.ok(events.some(e=>e.type==="message_end"&&e.message?.role==="user"&&JSON.stringify(e.message).includes(`${cwd}/literal`)));
  assert.ok(results.some(m=>m.isError&&JSON.stringify(m).includes("ENOENT")));
  assert.equal(await readFile(join(worker.dir,"workspace/out.txt"),"utf8"),"OK\n");
  await assert.rejects(readFile(join(cwd,"out.txt")),/ENOENT/);
  assert.equal(await readFile(outside,"utf8"),"outside-marker\n");
});
