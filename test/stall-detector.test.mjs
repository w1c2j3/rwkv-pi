import assert from 'node:assert/strict';
import test from 'node:test';
import { StallDetector, repeatedProse, stallLimits } from '../dist/stall-detector.js';

let id = 0;
function step(detector, name, args, text, error = false) {
  const call = { type: 'toolCall', id: String(++id), name, arguments: args };
  return detector.turn({ role: 'assistant', stopReason: 'toolUse', content: [call] }, [
    { role: 'toolResult', toolCallId: call.id, content: [{ type: 'text', text }], isError: error },
  ]);
}
const final = text => ({ role: 'assistant', stopReason: 'stop', content: [{ type: 'text', text }] });

test('detects repeated successful calls and multi-operation cycles by actual observations', () => {
  for (const width of [1, 2, 3, 6]) {
    const d = new StallDetector();
    for (let i = 0; i < width * 3; i++) {
      const result = step(d, 'read', { path: `${i % width}.txt` }, `contents-${i % width}`);
      if (i < width * 3 - 1) assert.equal(result, undefined);
      else assert.equal(result.code, width === 1 ? 'identical_call' : 'tool_cycle');
    }
  }
});
test('changing polling results and different source files are progress; a single failure can recover', () => {
  const d = new StallDetector();
  for (let i = 0; i < 20; i++) assert.equal(step(d, 'bash', { command: 'status' }, `progress ${i}`), undefined);
  assert.equal(step(d, 'read', { path: 'missing' }, 'ENOENT', true), undefined);
  assert.equal(step(d, 'read', { path: 'real' }, 'real contents'), undefined);
  assert.equal(d.turn(final('The missing file does not exist; real is present.'), []), undefined);
});
test('errors with changed arguments, early path abandonment, empty answers, and protocol failures', () => {
  const d = new StallDetector();
  step(d, 'read', { path: 'a' }, 'ENOENT', true);
  step(d, 'read', { path: 'b' }, 'ENOENT', true);
  assert.equal(step(d, 'read', { path: 'c' }, 'ENOENT', true).code, 'consecutive_errors');
  const path = new StallDetector();
  step(path, 'read', { path: '/wrong/path' }, 'ENOENT', true);
  assert.equal(path.turn(final('找不到这个文件，无法完成任务。'), []).code, 'gave_up_after_error');
  assert.equal(new StallDetector().turn(final(''), []).code, 'empty_answer');
  assert.equal(new StallDetector().turn({ role: 'assistant', stopReason: 'error', content: [], errorMessage: 'Invalid XML' }, []).code, 'model_error');
  assert.equal(new StallDetector().turn({ role: 'assistant', stopReason: 'aborted', content: [] }, []), undefined);
});
test('streamed prose repetition is detected, code repetition and short emphatic phrases are ignored', () => {
  const phrase = '我需要先检查当前目录，然后读取文件，再看看下一步应该怎么做。';
  assert.equal(repeatedProse(phrase.repeat(32)).code, 'repeated_text');
  assert.equal(repeatedProse('```js\n' + 'console.log("repeat");\n'.repeat(80) + '```'), undefined);
  assert.equal(repeatedProse('好好好。'), undefined);
});
test('budget limits terminate open tool loops but allow a final answer at the limit', () => {
  const limits = { ...stallLimits({}), rwkvTurns: 2 };
  const d = new StallDetector(limits);
  step(d, 'read', { path: 'a' }, 'a');
  assert.equal(step(d, 'read', { path: 'b' }, 'b').code, 'turn_budget');
  const good = new StallDetector(limits);
  step(good, 'read', { path: 'a' }, 'a');
  assert.equal(good.turn(final('Done.'), []), undefined);
  assert.throws(() => stallLimits({ RWKV_COLLAB_MAX_TURNS: '0' }), /positive/);
});
