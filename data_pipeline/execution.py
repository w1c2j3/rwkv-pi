"""Explicit native-Pi collection; keep one budget ledger and owner lock per account."""
from pathlib import Path
import fcntl
import os
import subprocess
from .products import read, write, profile_identity, load_questions


def collect_plan(root, profile_paths):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    assert plan['state'] == 'planned', 'An existing attempt must not be replayed'
    assert load_questions(plan['questionsRoot'])['questionSetId'] == plan['questionSetId']
    profiles = [(Path(path).resolve(), read(path)) for path in profile_paths]
    keys = [profile_identity(p)[0] for _, p in profiles]
    assert keys == [t['profileId'] for t in plan['teachers']], 'Use the exact planned teacher profiles'
    for _, p in profiles:
        budget, ex = p['budget'], p['execution']
        assert budget['cnyLimit'] > 0 and 0 < budget['incrementCny'] <= budget['cnyLimit']
        assert Path(budget['ownerLock']).parent.is_dir()
        for name in ('node', 'launcher', 'budgetModule', 'harnessModule'):
            assert Path(ex[name]).is_file(), 'Missing collection dependency: ' + name
        assert Path(p['credentialFile']).is_file()
    results = []
    plan.update(state='collection_running', authorizedToStart=True)
    write(root / 'plan.json', plan)
    try:
        for (path, profile), key in zip(profiles, keys):
            ex, budget = profile['execution'], profile['budget']
            env = dict(os.environ, RWKV_PI_SOURCE=ex['piSource'])
            if ex.get('studentConfig'):
                student = read(ex['studentConfig'])
                env.update(RWKV_BACKEND='lightning', RWKV_BASE_URL=student['baseUrl'],
                           RWKV_MODEL=student['modelId'], RWKV_API_KEY=student['apiKey'],
                           RWKV_CONTEXT_WINDOW=str(student['contextWindow']), RWKV_MAX_TOKENS=str(student['maxTokens']),
                           RWKV_TOOL_FORMAT='xml', RWKV_PROMPT_TEMPLATE='xml', RWKV_TOOL_CALL_RECOVERY='1',
                           RWKV_IDENTICAL_CALL_LIMIT='2', RWKV_PLUGINS='none',
                           RWKV_EDIT_MODE=str(student.get('editMode', profile.get('editMode', 'line'))))
                if student.get('adapterId'):
                    env['RWKV_ADAPTER_ID'] = str(student['adapterId'])
                if student.get('adapterVersion'):
                    env['RWKV_ADAPTER_VERSION'] = str(student['adapterVersion'])
                if student.get('adapterScale') is not None:
                    env['RWKV_ADAPTER_SCALE'] = str(student['adapterScale'])
                if student.get('historyMode'):
                    env['RWKV_HISTORY_MODE'] = str(student['historyMode'])
            env['PATH'] = str(Path(ex['node']).parent) + ':' + env.get('PATH', '')
            with Path(budget['ownerLock']).open('a+') as lock:
                # Existing production owns this same lock. Fail visibly instead of
                # launching a second writer or waiting silently for hours.
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                proc = subprocess.run([ex['node'], ex['launcher'],
                                       str(Path(__file__).with_name('collect.mjs')), str(root), str(path), key],
                                      cwd=ex['runtime'], env=env)
            results.append({'profileId': key, 'exitCode': proc.returncode})
            if proc.returncode:
                break
    finally:
        complete = len(results) == len(profiles) and all(r['exitCode'] == 0 for r in results)
        plan.update(state='collected_awaiting_review' if complete else 'collection_stopped',
                    collections=results, trainingRows=0)
        write(root / 'plan.json', plan)
    return plan
