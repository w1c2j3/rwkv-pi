// Protocol tests with synthetic responses; never model-quality or training data.
import assert from 'node:assert/strict';
import test from 'node:test';
import {createModels} from '@earendil-works/pi-ai';
import { readStepConfig, createStepProvider, withChineseReasoning, CHINESE_REASONING_SUFFIX } from '../dist/step-teacher.js';
import extension from '../dist/step-extension.js';

const env = { STEP_API_KEY:'test-key-no-network' };
test('Step billing/model selection is explicit and preserves original thinking mode', () => {
  const base=readStepConfig(env);
  assert.equal(base.modelId,'step-3.5-flash');
  assert.equal(base.baseUrl,'https://api.stepfun.com/v1');
  assert.equal(base.maxTokens,16384);
  assert.equal(base.reasoningEffort,undefined);
  assert.equal(readStepConfig({...env,STEP_BILLING:'plan'}).baseUrl,'https://api.stepfun.com/step_plan/v1');
  assert.equal(readStepConfig({...env,STEP_MODEL:'step-3.5-flash-2603'}).reasoningEffort,'high');
  assert.equal(readStepConfig({...env,STEP_MODEL:'step-3.7-flash'}).reasoningEffort,'high');
  assert.equal(readStepConfig({...env,STEP_MODEL:'step-3.7-flash',STEP_REASONING_EFFORT:'medium'}).reasoningEffort,'medium');
  const step5=readStepConfig({...env,STEP_MODEL:'step-5-preview',STEP_REASONING_EFFORT:'low',STEP_BILLING:'plan'});
  assert.equal(step5.modelId,'step-5-preview'); assert.equal(step5.contextWindow,1000000);
  assert.equal(step5.reasoningEffort,'low'); assert.equal(step5.baseUrl,'https://api.stepfun.com/step_plan/v1');
  assert.throws(()=>readStepConfig({...env,STEP_MODEL:'step-5-preview',STEP_CONTEXT_WINDOW:'1000001'}),/1000000/);
  assert.throws(()=>readStepConfig({...env,STEP_MODEL:'step-3.5-flash-2603',STEP_REASONING_EFFORT:'medium'}),/not medium/);
  assert.throws(()=>readStepConfig({...env,STEP_REASONING_EFFORT:'low'}),/only/);
  assert.throws(()=>readStepConfig({...env,STEP_REASONING_EFFORT:'off'}),/thinking stays enabled/);
  assert.throws(()=>readStepConfig({TEACHER_API_KEY:'must-not-reuse'}),/STEP_API_KEY/);
  assert.throws(()=>readStepConfig({...env,STEP_BILLING:'auto'}),/api or plan/);
});

test('Chinese instruction transforms the user message once and retains attachments', async () => {
  const saved=process.env.STEP_API_KEY;process.env.STEP_API_KEY=env.STEP_API_KEY;
  let hook;
  try { extension({registerProvider(){},on(event,fn){ assert.equal(event,'input');hook=fn; }}); }
  finally {if(saved===undefined)delete process.env.STEP_API_KEY;else process.env.STEP_API_KEY=saved;}
  const images=[{type:'image',data:'fixture',mimeType:'image/png'}];
  const out=await hook({text:'检查当前目录。',images},{model:{provider:'stepfun'}});
  assert.equal(out.action,'transform');assert.equal(out.images,images);
  assert.equal(out.text,'检查当前目录。\n\n'+CHINESE_REASONING_SUFFIX);
  assert.equal(withChineseReasoning(out.text),out.text);
  assert.deepEqual(await hook({text:'unchanged'},{model:{provider:'teacher'}}),{action:'continue'});
});

test('real pi codec preserves system/tools, Chinese reasoning and tool-result continuation', async () => {
  const provider=createStepProvider(readStepConfig(env));
  const model=provider.getModels()[0];
  const models=createModels(); models.setProvider(provider);
  const system='Native pi system fixture: retain this byte for byte.';
  const tools=[{name:'read',description:'Read file',parameters:{type:'object',properties:{path:{type:'string'}},required:['path']}}];
  const messages=[{role:'user',content:withChineseReasoning('读取 README.md。'),timestamp:Date.now()}];
  const payloads=[];
  let fetchError;
  const fakeFetch=async(url,init)=>{
    try {
    assert.equal(new URL(url).href,'https://api.stepfun.com/v1/chat/completions');
    const body=JSON.parse(init.body);payloads.push(body);
    assert.equal(body.messages[0].role,'system');assert.equal(body.messages[0].content,system);
    assert.equal(body.tools[0].function.name,'read');
    assert.deepEqual(body.tools[0].function.parameters,tools[0].parameters);
    assert.equal(body.reasoning_format,'deepseek-style');
    assert.equal(body.reasoning_effort,undefined);assert.equal(body.chat_template_kwargs,undefined);
    const deltas=payloads.length===1
      ? [{reasoning_content:'先读取文件确认内容。'},{tool_calls:[{index:0,id:'call_step1',type:'function',function:{name:'read',arguments:'{"path":"README.md"}'}}]}]
      : [{reasoning_content:'已收到实际工具结果。'},{content:'README.md 内容为你好。'}];
    const frames=deltas.map(delta=>({choices:[{index:0,delta,finish_reason:null}]}));
    frames.push({choices:[{index:0,delta:{},finish_reason:payloads.length===1?'tool_calls':'stop'}],usage:{prompt_tokens:50,completion_tokens:20,total_tokens:70}});
    return new Response(frames.map(f=>'data: '+JSON.stringify(f)+'\n\n').join('')+'data: [DONE]\n\n',{headers:{'content-type':'text/event-stream'}});
    } catch(error) { fetchError=error;throw error; }
  };
  const first=await models.streamSimple(model,{systemPrompt:system,messages,tools},{reasoning:'high',apiKey:env.STEP_API_KEY,fetch:fakeFetch}).result();
  if(fetchError)throw fetchError;
  assert.equal(first.stopReason,'toolUse',first.errorMessage);
  assert.equal(first.content.find(b=>b.type==='thinking').thinking,'先读取文件确认内容。');
  const call=first.content.find(b=>b.type==='toolCall'); assert.deepEqual(call.arguments,{path:'README.md'});
  messages.push(first,{role:'toolResult',toolCallId:call.id,toolName:'read',content:[{type:'text',text:'你好'}],isError:false,timestamp:Date.now()});
  const last=await models.streamSimple(model,{systemPrompt:system,messages,tools},{reasoning:'high',apiKey:env.STEP_API_KEY,fetch:fakeFetch}).result();
  assert.equal(last.stopReason,'stop');assert.match(last.content.find(b=>b.type==='text').text,/你好/);
  assert.equal(payloads[1].messages.at(-1).role,'tool');
  assert.equal(payloads[1].messages.find(m=>m.role==='assistant').reasoning_content,'先读取文件确认内容。');
});
