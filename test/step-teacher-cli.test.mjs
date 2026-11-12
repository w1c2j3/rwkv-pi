// Real CLI + mock transport only. Validates input transformation, not model quality.
import assert from 'node:assert/strict';
import test from 'node:test';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdtemp,writeFile,readFile,rm,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {CHINESE_REASONING_SUFFIX} from '../dist/step-teacher.js';
const run=promisify(execFile);
test('Step CLI sends the native seven-tool prompt and appends Chinese instructions to user input', {timeout:60000}, async t=>{
 const root=await mkdtemp(join(tmpdir(),'step-cli-fixture-'));
 t.after(()=>rm(root,{recursive:true,force:true}));
 const ws=join(root,'workspace');await mkdir(ws);
 const preload=join(root,'fetch.mjs');
 await writeFile(preload,`
 import {writeFileSync} from 'node:fs';
 const fakeFetch=async (url,init)=>{
  if(String(url)!=='https://api.stepfun.com/v1/chat/completions')throw Error('Fixture blocks all other network');
  writeFileSync(process.env.STEP_FIXTURE_CAPTURE,init.body);
  const frames=[{reasoning_content:'先检查中文输入。'},{content:'中文接入可用。'}].map(delta=>({choices:[{index:0,delta,finish_reason:null}]}));
  frames.push({choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,total_tokens:20}});
  return new Response(frames.map(f=>'data: '+JSON.stringify(f)+'\\n\\n').join('')+'data: [DONE]\\n\\n',{headers:{'content-type':'text/event-stream'}});
 };
 Object.defineProperty(globalThis,'fetch',{get:()=>fakeFetch,set:()=>{},configurable:false});
 `);
 const capture=join(root,'request.json');
 const env={...process.env,STEP_API_KEY:'fixture-no-network',STEP_MODEL:'step-3.5-flash',STEP_BILLING:'api',STEP_REASONING_EFFORT:'',
  PI_CODING_AGENT_DIR:join(root,'agent'),RWKV_PI_SOURCE:resolve('.upstream/pi'),STEP_FIXTURE_CAPTURE:capture,
  NODE_OPTIONS:`--import=${pathToFileURL(preload).href}`};
 const execution=run(process.execPath,[resolve('dist/step-cli.js'),'--offline','--no-approve','--no-session','--no-themes','--print','请确认中文接入。'],{cwd:ws,env,timeout:50000});
 execution.child.stdin.end();
 const {stdout}=await execution;
 assert.match(stdout,/中文接入可用/);
 const request=JSON.parse(await readFile(capture,'utf8'));
 assert.equal(request.model,'step-3.5-flash');
 assert.deepEqual(request.tools.map(t=>t.function.name),['read','write','edit','bash','grep','find','ls']);
 assert.equal(request.messages[0].role,'system');
 assert.ok(request.messages[0].content.length>1500);
 assert.ok(!request.messages[0].content.includes(CHINESE_REASONING_SUFFIX));
 const content=request.messages.find(m=>m.role==='user').content;
 const text=typeof content==='string'?content:content.filter(b=>b.type==='text').map(b=>b.text).join('');
 assert.equal(text,'请确认中文接入。\n\n'+CHINESE_REASONING_SUFFIX);
});
