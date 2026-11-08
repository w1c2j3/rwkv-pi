// Training-only packing: canonical online history is never given an extra EOS.
import {renderPrompt} from '../dist/prompt.js';
export function normalizeReasoning(message) {
  const m=structuredClone(message);
  if(m.reasoning_details!==undefined){
    if(!Array.isArray(m.reasoning_details)||!m.reasoning_details.every(d=>d.type==='reasoning.text'&&typeof d.text==='string'))throw Error('Unsupported reasoning details');
    const value=m.reasoning_details.map(d=>d.text).join('');
    for(const key of ['reasoning','reasoning_content'])if(m[key]!=null&&m[key]!==value)throw Error('Conflicting reasoning');
    m.reasoning_content=value;delete m.reasoning;delete m.reasoning_details;
  }
  return m;
}
export function packTrajectory(samples,format='xml') {
  const spans=[];let span;
  for(const sample of samples){
    const messages=sample.messages.map(normalizeReasoning);
    const prefix=renderPrompt(messages.slice(0,-1),sample.tools,true,true,format);
    const full=renderPrompt(messages,sample.tools,true,false,format);
    if(!full.startsWith(prefix)||full.length===prefix.length)throw Error('Invalid continuation');
    if(!span||!prefix.startsWith(span.text)){
      span={segments:[],text:'',decisions:[],resetReason:span?'request_history_changed':'trajectory_start'};spans.push(span);
    }
    const gap=prefix.slice(span.text.length);
    if(!gap)throw Error('No untrained boundary between decisions');
    span.segments.push({text:gap,train:false},{text:full.slice(prefix.length),train:true,end_target:0});
    span.text=full;span.decisions.push({id:sample.id,step:sample.step,prefixChars:prefix.length,endChars:full.length});
  }
  for(const span of spans)span.segments.push({text:'<EOD>',train:false});
  return spans;
}
// Mirrors the training loader contract; labels are indexed like next tokens.
export function tokenLayout(segments,encoded){
  const tokens=[],labels=[],train=[],overrides=[];
  for(let i=0;i<segments.length;i++){
    const s=segments[i],ids=encoded[i];
    if(!ids.length)throw Error('Empty token segment');
    tokens.push(...ids);labels.push(...ids);train.push(...ids.map(()=>s.train?1:0));
    if(s.end_target!==undefined){if(!s.train||s.end_target!==0)throw Error('Invalid EOS override');overrides.push(tokens.length);}
  }
  for(const index of overrides){
    if(index>=tokens.length||train[index])throw Error('EOS override collides with assistant supervision');
    labels[index]=0;train[index]=1;
  }
  return {tokens,labels,train,overrides};
}
