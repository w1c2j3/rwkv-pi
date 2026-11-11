// Native pi codec, synthetic SSE only: these are protocol tests, not model-quality evidence.
import assert from 'node:assert/strict';
import test from 'node:test';
import {createModels} from '@earendil-works/pi-ai';
import {readTeacherConfig,createTeacherProvider} from '../dist/teacher.js';

const env={TEACHER_KIND:'deepseek',TEACHER_MODEL:'deepseek-flash',TEACHER_BASE_URL:'https://api.deepseek.com',
  TEACHER_API_KEY:'fixture-not-a-key',TEACHER_MAX_TOKENS:'16384'};

test('DeepSeek is opt-in; local Qwen parameters are preserved',()=>{
  const qwen=createTeacherProvider(readTeacherConfig({TEACHER_MODEL:'local-qwen',TEACHER_BASE_URL:'http://localhost:8000/v1'})).getModels()[0];
  assert.equal(qwen.samplingParams,undefined);
  assert.equal(qwen.compat.requiresReasoningContentOnAssistantMessages,undefined);
  const ds=createTeacherProvider(readTeacherConfig(env)).getModels()[0];
  assert.deepEqual(ds.samplingParams.thinking,{type:'enabled'});
  assert.equal(ds.samplingParams.reasoning_effort,'high');
  assert.throws(()=>readTeacherConfig({...env,TEACHER_BASE_URL:'https://other.invalid'}),/official/);
  assert.throws(()=>readTeacherConfig({...env,TEACHER_REASONING_EFFORT:'off'}),/stays enabled/);
});

test('DeepSeek preserves every historical reasoning field across two actual codec tool turns',async()=>{
  const p=createTeacherProvider(readTeacherConfig(env)), models=createModels();models.setProvider(p);
  const model=p.getModels()[0], payloads=[];
  const context={systemPrompt:'Native pi system fixture.',messages:[{role:'user',content:'先读 a，再读 b。请用中文分析。',timestamp:Date.now()}],
    tools:[{name:'read',description:'Read file',parameters:{type:'object',properties:{path:{type:'string'}},required:['path']}}]};
  const fakeFetch=async(url,init)=>{
    const body=JSON.parse(init.body);payloads.push(body);const n=payloads.length;
    assert.equal(url,'https://api.deepseek.com/chat/completions');
    assert.equal(body.messages[0].content,context.systemPrompt);
    assert.equal(body.thinking.type,'enabled');assert.equal(body.reasoning_effort,'high');
    const delta=n<3?{reasoning_content:`中文思考${n}。`,tool_calls:[{index:0,id:`c${n}`,type:'function',function:{name:'read',arguments:JSON.stringify({path:n===1?'a':'b'})}}]}:
      {reasoning_content:'已根据两个文件核对。',content:'核对完毕。'};
    const frames=[{choices:[{index:0,delta,finish_reason:null}]},{choices:[{index:0,delta:{},finish_reason:n<3?'tool_calls':'stop'}],usage:{prompt_tokens:100,completion_tokens:20}}];
    return new Response(frames.map(x=>'data: '+JSON.stringify(x)+'\n\n').join('')+'data: [DONE]\n\n',{headers:{'content-type':'text/event-stream'}});
  };
  for(let n=1;n<=3;n++){
    const reply=await models.streamSimple(model,context,{reasoning:'high',fetch:fakeFetch}).result();
    assert.equal(reply.stopReason,n<3?'toolUse':'stop',reply.errorMessage);context.messages.push(reply);
    if(n<3)context.messages.push({role:'toolResult',toolCallId:`c${n}`,toolName:'read',content:[{type:'text',text:`文件${n}`}],isError:false,timestamp:Date.now()});
  }
  assert.deepEqual(payloads[2].messages.filter(m=>m.role==='assistant').map(m=>m.reasoning_content),['中文思考1。','中文思考2。']);
});
