import test from 'node:test';
import assert from 'node:assert/strict';
import {delegationTools, delegationEligibility, assertPublicHttpUrl, createProductionWebTools} from './delegation.mjs';

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

test('multi-turn delegated followups require all turns completed', () => {
  const e = evidence();
  e.conversation = {allTurnsCompleted: true, turns: [{index: 1, userPrompt: 'turn 1', completed: true}, {index: 2, userPrompt: 'turn 2', completed: true}]};
  assert.equal(delegationEligibility(e).passed, true);
  e.conversation = {allTurnsCompleted: false, turns: [{index: 1, userPrompt: 'turn 1', completed: true}, {index: 2, userPrompt: 'turn 2', completed: false}]};
  assert.equal(delegationEligibility(e).passed, false);
  assert.ok(delegationEligibility(e).reasons.includes('incomplete_followups'));
});

test('production web tools enforce public URL safety, budgets, and evidence framing', async () => {
  assert.equal(assertPublicHttpUrl('https://example.com/docs?q=1'), 'https://example.com/docs?q=1');
  assert.throws(() => assertPublicHttpUrl('http://127.0.0.1:8000/v1'), /Public HTTP/);
  assert.throws(() => assertPublicHttpUrl('http://localhost/admin'), /Public HTTP/);
  assert.throws(() => assertPublicHttpUrl('https://user:pass@example.com/'), /Public HTTP/);
  assert.throws(() => assertPublicHttpUrl('http://192.168.1.1/status'), /Public HTTP/);
  const calls = [];
  const tools = createProductionWebTools({
    maxCalls: 2,
    recordEvent: e => calls.push(e),
    callWeb: async (kind, body) => kind === 'search'
      ? {results: [{title: 'T', url: 'https://example.com/a', date: '2026-11-10', snippet: 'S', summary: 'M'}]}
      : {title: 'Page', content: 'x'.repeat(15000)},
  });
  const search = tools.find(t => t.name === 'web_search');
  const fetchTool = tools.find(t => t.name === 'web_fetch');
  const r1 = JSON.parse((await search.execute('c1', {query: 'RWKV'})).content[0].text);
  assert.equal(r1.externalUntrustedData, true);
  assert.equal(r1.results[0].url, 'https://example.com/a');
  const r2 = JSON.parse((await fetchTool.execute('c2', {url: 'https://example.com/a'})).content[0].text);
  assert.equal(r2.truncated, true);
  assert.equal(calls.length, 2);
  await assert.rejects(() => search.execute('c3', {query: 'third'}), /Web tool call budget reached/);
});
