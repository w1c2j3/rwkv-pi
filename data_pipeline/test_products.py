"""Synthetic engineering records only; no teacher calls or real training assets."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from .products import (digest, encoded, write, read, export_questions, load_questions,
                       plan_trajectories, profile_identity, export_training, compare_attempts)


class ProductTests(unittest.TestCase):
    def setUp(self):
        parent = Path('/mnt/f/rwkv-pi-data/artifacts/data-pipeline-tests')
        if not Path('/mnt/f').is_mount():
            self.fail('F: must be mounted')
        parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.bundle = {'task': {'schemaVersion': 1, 'source': 'synthetic-engineering-only', 'id': 'engineering-only', 'family': 'test-only', 'split': 'train',
                               'prompt': 'Engineering fixture, not a real question.',
                               'tools': ['read'], 'files': {'fixture.txt': 'synthetic'},
                               'checks': [{'type': 'file_equals', 'path': 'fixture.txt', 'expected': 'synthetic'}]},
                       'followups': [{'prompt': 'Synthetic continuation.'}],
                       'review': {'semanticReview': 'pending'}}
        write(self.source / 'one.json', self.bundle)
        self.questions = self.root / 'questions'
        self.qm = export_questions([self.source], self.questions)
        self.profile = {'provider': 'engineering', 'model': 'fixture-a', 'baseUrl': 'http://127.0.0.1:1',
                        'api': 'openai-completions', 'thinking': 'high', 'maxTokens': 256,
                        'contextWindow': 4096, 'protocol': 'canonical XML+JSON',
                        'appendSystemPrompt': 'Test instruction', 'promptSuffix': '\nSynthetic constraint.'}

    def entry(self, name, model=None):
        folder = self.root / name
        folder.mkdir()
        profile = {**self.profile, 'model': model or self.profile['model']}
        pk, _ = profile_identity(profile)
        qh = self.qm['questions'][0]['questionSha256']
        run = {'runId': name, 'task': self.bundle['task'], 'teacherProfileId': pk, 'questionSha256': qh,
               'model': {'provider': profile['provider'], 'id': profile['model']}, 'role': 'teacher',
               'status': 'passed', 'checks': [{'passed': True}],
               'messages': [{'role': 'assistant', 'provider': profile['provider'], 'model': profile['model']}],
               'appendSystemPrompt': profile['appendSystemPrompt'], 'promptSuffix': profile['promptSuffix'],
               'generation': {'thinkingLevel': 'high', 'maxTokens': 256, 'contextWindow': 4096},
               'conversation': {'allTurnsCompleted': True,
                                'turns': [{'userPrompt': self.bundle['task']['prompt']}, {'userPrompt': self.bundle['followups'][0]['prompt']}]}}
        raw = encoded(run)
        review = {'verdict': 'pass', 'reviewer': 'Synthetic test', 'notes': 'Not a real quality approval',
                  'runSha256': digest(raw), 'questionSha256': qh, 'profileId': pk, 'effectiveToolCalls': 20,
                  'semanticPassed': True, 'wirePassed': True, 'scopePassed': True, 'noPadding': True}
        row = encoded({'segments': [{'text': 'Synthetic prefix', 'train': False},
                                   {'text': 'Synthetic target ' + name, 'train': True, 'end_target': 0}]})
        pm = {'format': 'rwkv-lightning-segments-eos-target-v2', 'requiredTrainingCapability': 'segment-end-target-v1',
              'promptVersion': 'rwkv-pi-xml-continuation-v3', 'toolFormat': 'xml',
              'engineRevision': 'test-only', 'vocabSha256': 'test-only',
              'files': {'train.jsonl': digest(row)}, 'sources': [{'runId': name, 'sha256': digest(raw)}],
              'rows': [{'runId': name, 'split': 'train', 'family': 'test-only'}]}
        for filename, value in [('profile.json', profile), ('run.json', run), ('review.json', review), ('manifest.json', pm)]:
            write(folder / filename, value)
        (folder / 'train.jsonl').write_bytes(row)
        return {k: str(folder / v) for k, v in [('profile', 'profile.json'), ('run', 'run.json'), ('review', 'review.json'), ('packedDataset', '')]}

    def test_question_export_preserves_bytes_and_never_runs_checks(self):
        self.assertEqual(self.qm['mode'], 'questions')
        self.assertEqual(self.qm['trainingRows'], 0)
        self.assertEqual((self.questions / 'bundles/engineering-only.json').read_bytes(), (self.source / 'one.json').read_bytes())
        self.assertEqual(load_questions(self.questions)['questionCount'], 1)

    def test_same_id_revision_conflict_and_snapshot_change(self):
        changed = copy.deepcopy(self.bundle)
        changed['task']['environment'] = {'image': 'engineering-only'}
        write(self.source / 'two.json', changed)
        with self.assertRaisesRegex(AssertionError, 'Conflicting revisions'):
            export_questions([self.source], self.root / 'conflict')
        write(self.questions / 'bundles/engineering-only.json', changed)
        with self.assertRaisesRegex(AssertionError, 'snapshot changed'):
            load_questions(self.questions)

    def test_two_teachers_one_frozen_question_and_no_credential_export(self):
        profile = {**self.profile, 'credentialFile': '/private/test.key', 'apiKey': 'not-a-real-credential'}
        second = {**profile, 'model': 'fixture-b'}
        plan = plan_trajectories(self.questions, [profile, second], self.root / 'planned')
        self.assertEqual(plan['questionCount'], 1)
        self.assertEqual(len(plan['teachers']), 2)
        a, b = plan['teachers']
        self.assertNotEqual(a['jobs'][0]['jobId'], b['jobs'][0]['jobId'])
        self.assertEqual(a['jobs'][0]['questionSha256'], b['jobs'][0]['questionSha256'])
        self.assertNotIn('credential', json.dumps(plan))
        self.assertFalse(plan['authorizedToStart'])
        self.assertEqual(profile_identity(profile)[0], profile_identity(self.profile)[0])

    def test_profile_changes_and_credentials_in_endpoint(self):
        a = profile_identity(self.profile)[0]
        self.assertNotEqual(a, profile_identity({**self.profile, 'thinking': 'off'})[0])
        with self.assertRaisesRegex(AssertionError, 'credentials'):
            profile_identity({**self.profile, 'baseUrl': 'https://test.invalid?key=private'})

    def test_training_separates_teachers_and_preserves_exact_rows(self):
        entries = [self.entry('run-a'), self.entry('run-b', 'fixture-b')]
        output = self.root / 'training'
        report = export_training(self.questions, entries, output)
        self.assertEqual((report['questionCount'], report['trajectoryCount'], report['trainingRows']), (1, 2, 2))
        self.assertEqual(len(report['teachers']), 2)
        for teacher in report['teachers']:
            source = entries[0] if teacher['teacher']['model'] == 'fixture-a' else entries[1]
            self.assertEqual((output / teacher['path'] / 'train.jsonl').read_bytes(),
                             (Path(source['packedDataset']) / 'train.jsonl').read_bytes())
            self.assertEqual(read(output / teacher['path'] / 'manifest.json')['requiredTrainingCapability'], 'segment-end-target-v1')

    def test_pending_review_short_trajectory_or_stale_run_not_exported(self):
        entry = self.entry('run-test')
        original = read(entry['review'])
        for change in [{'verdict': 'pending'}, {'effectiveToolCalls': 19}, {'noPadding': False}, {'runSha256': 'old'}]:
            with self.subTest(change=change):
                write(entry['review'], {**original, **change})
                with self.assertRaises(AssertionError):
                    export_training(self.questions, [entry], self.root / 'forbidden')
                self.assertFalse((self.root / 'forbidden').exists())

    def test_changed_followup_or_teacher_cannot_borrow_existing_review(self):
        entry = self.entry('run-test')
        original = read(entry['run'])
        for change in ['followup', 'teacher', 'profile']:
            modified = copy.deepcopy(original)
            if change == 'followup':modified['conversation']['turns'][1]['userPrompt'] = 'Different request'
            if change == 'teacher':modified['messages'][0]['model'] = 'different-teacher'
            if change == 'profile':modified['teacherProfileId'] = 'old'
            write(entry['run'], modified)
            with self.subTest(change=change), self.assertRaises(AssertionError):
                export_training(self.questions, [entry], self.root / 'forbidden')

    def test_comparison_never_groups_different_question_versions(self):
        report = compare_attempts([{'questionId': 'test', 'questionSha256': 'a', 'profileId': 'model-a'},
                                   {'questionId': 'test', 'questionSha256': 'a', 'profileId': 'model-b'},
                                   {'questionId': 'test', 'questionSha256': 'b', 'profileId': 'model-a'}])
        self.assertEqual(report['questionVersions'], 2)
        self.assertEqual(len(report['questions'][0]['attempts']), 2)

    def test_cli_question_mode_rejects_execution(self):
        proc = subprocess.run([sys.executable, '-B', '-m', 'data_pipeline', 'produce', '--mode', 'questions',
                               '--source', str(self.source), '--out', str(self.root / 'forbidden'), '--execute'],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.root / 'forbidden').exists())


if __name__ == '__main__':
    unittest.main()
