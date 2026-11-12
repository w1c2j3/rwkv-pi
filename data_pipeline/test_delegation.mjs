import test from 'node:test';
import assert from 'node:assert/strict';
import {delegationTools, delegationEligibility} from './delegation.mjs';

const evidence = () => ({agents: [{state:'completed', requests:3, toolCalls:2, finishedAt:1, write:true, applied:true}],
  messages:[{role:'assistant', provider:'deepseek', model:'deepseek-flash', stopReason:'stop'}],
  events:['subagent_plan','subagent_start','subagent_wait','subagent_diff','subagent_apply','subagent_verify'].map(name=>({name})),
  checks:[{passed:true}], teacher:{provider:'deepseek',id:'deepseek-flash'}});

test('a successful parent without real child execution cannot be exported', () => {
  const e = evidence(); e.agents[0].requests=0;
  assert.equal(delegationEligibility(e).passed,false);
});
test('child success cannot replace independent final acceptance', () => {
  const e=evidence();e.checks=[{passed:false}];
  assert.equal(delegationEligibility(e).passed,false);
});
test('student responses cannot be relabeled as teacher targets', () => {
  const e=evidence();e.messages[0].provider='rwkv';
  assert.equal(delegationEligibility(e).passed,false);
});
test('an applied patch must be reviewed again when its diff changes', async () => {
  let diff='first', applied=0;
  const manager={list:()=>[{finishedAt:1}],diff:()=>diff,apply:()=>{applied++;return {};}};
  const ts=delegationTools({manager,plan:{},verify:()=>[],recordEvent:()=>{}});
  const get=n=>ts.find(t=>t.name===n);
  await assert.rejects(get('subagent_apply').execute('a',{id:'worker'}));
  await get('subagent_diff').execute('b',{id:'worker'});diff='changed';
  await assert.rejects(get('subagent_apply').execute('c',{id:'worker'}));
  await get('subagent_diff').execute('d',{id:'worker'});
  await get('subagent_apply').execute('e',{id:'worker'});assert.equal(applied,1);
});
test('complete delegation remains subject to semantic review', () => {
  const e=delegationEligibility(evidence());assert.equal(e.passed,true);assert.equal(e.semanticReviewRequired,true);
});

test('an explicitly selected different teacher remains independent', () => {
 const e=evidence();e.teacher={provider:'test-provider',id:'test-model'};e.messages[0].provider='test-provider';e.messages[0].model='test-model';
 assert.equal(delegationEligibility(e).passed,true);e.messages.push({...e.messages[0],model:'other'});assert.equal(delegationEligibility(e).passed,false);
});
