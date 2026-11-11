import assert from 'node:assert/strict';
import test from 'node:test';
import { resolvePlugins } from '../dist/plugins.js';

test('plugins are independent opt-ins; teacher settings do not enable them', () => {
  assert.deepEqual(resolvePlugins([], { TEACHER_MODEL: 'configured' }).enabled, []);
  assert.deepEqual(resolvePlugins([], { RWKV_PLUGINS: 'subagents' }).enabled, ['subagents']);
  assert.deepEqual(resolvePlugins([], { RWKV_PLUGINS: 'strong-model' }).enabled, ['strong-model']);
  assert.deepEqual(resolvePlugins([], { RWKV_PLUGINS: 'strong-model,subagents' }).enabled, ['strong-model', 'subagents']);
});

test('explicit plugin switches override defaults and preserve pi arguments and literal prompts', () => {
  const env = { RWKV_PLUGINS: 'strong-model,subagents' };
  assert.deepEqual(resolvePlugins(['--plugins', 'none', '--offline'], env), { enabled: [], args: ['--offline'] });
  assert.deepEqual(resolvePlugins(['--no-collab'], env).enabled, ['subagents']);
  assert.deepEqual(resolvePlugins(['--no-subagents'], env).enabled, ['strong-model']);
  assert.deepEqual(resolvePlugins(['--collab', '--subagents', '--plugins=none', '--subagents'], {}).enabled, ['subagents']);
  assert.deepEqual(resolvePlugins(['--', '--collab'], {}), { enabled: [], args: ['--', '--collab'] });
  for (const args of [['--plugins'], ['--plugins', '--print'], ['--plugins='], ['--plugins=typo'], ['--plugins=none,subagents']]) {
    assert.throws(() => resolvePlugins(args, {}), /plugins/);
  }
});
