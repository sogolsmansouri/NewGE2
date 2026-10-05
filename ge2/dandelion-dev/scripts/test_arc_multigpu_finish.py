import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from arc_job_support import write_json
from run_arc_multigpu_campaign import multigpu_config, multigpu_flags
from run_arc_multigpu_finish import (apply_engine_override, apply_final_cohort, completed_case, freeze_retry, restart_base,
                                     source_campaign, validate_reference_contract)
from run_arc_paper_case import digest


class MultiGpuFinishTests(unittest.TestCase):
    def test_engine_override_is_optional(self):
        manifest = {'commit': 'unchanged'}
        self.assertIs(apply_engine_override(Path('/unused'), manifest, {}), manifest)

    def test_engine_override_rejects_unpinned_or_nonlocal_builds(self):
        for directory in ('relative', '/tmp/engine', '/mnt/local/smansou2/a/b'):
            with self.assertRaisesRegex(ValueError, 'node-local'):
                apply_engine_override(Path('/unused'), {},
                    {'engine_override': {'directory': directory}})
        with patch('run_arc_multigpu_finish.digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                apply_engine_override(Path('/unused'), {}, {'engine_override':
                    dict(directory='/mnt/local/smansou2/fixed', identity_sha256='pinned')})

    def test_engine_override_checks_sources_and_binaries_before_repointing(self):
        engine = Path('/mnt/local/smansou2/fixed')
        identity = dict(commit='new', source_comparison_status='pass',
            source_hashes={'native.cpp': 'source'}, engine_hashes={'gege_train': 'train', 'libge2.so': 'lib'})
        metadata = {'engine_override': dict(directory=str(engine), identity_sha256='identity', commit='new')}
        hashes = {str(engine/'build_identity.json'): 'identity', str(engine/'repo/native.cpp'): 'source',
                  str(engine/'build_git/gege_train'): 'train', str(engine/'build_git/libge2.so'): 'lib'}
        for corrupt in (None, 'source', 'train', 'lib'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                (base/'engine').symlink_to('/old/engine')
                with patch('pathlib.Path.read_text', return_value=json.dumps(identity)), patch(
                        'run_arc_multigpu_finish.subprocess.check_output', return_value='new\n'), patch(
                        'run_arc_multigpu_finish.subprocess.run'), patch(
                        'run_arc_multigpu_finish.digest', side_effect=lambda path:
                            'changed' if hashes[str(path)] == corrupt else hashes[str(path)]):
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, 'changed'):
                            apply_engine_override(base, {}, metadata)
                        self.assertEqual((base/'engine').readlink(), Path('/old/engine'))
                    else:
                        result = apply_engine_override(base, {'ge2_library_sha256': 'original'}, metadata)
                        self.assertEqual(result['commit'], 'new')
                        self.assertEqual(result['engine_hashes'], identity['engine_hashes'])
                        self.assertEqual(result['ge2_library_sha256'], 'original')
                        self.assertEqual((base/'engine').readlink(), engine)

    def test_completion_is_optional_and_requires_evidence_directory(self):
        self.assertIsNone(completed_case({}, 'payload', 'case'))
        with self.assertRaisesRegex(ValueError, 'evidence directory'):
            completed_case({'completion_root': '/tmp/receipts'}, 'payload', 'case')

    def test_completed_receipt_rejects_control_results_and_changed_evidence(self):
        root = Path('/home/smansou2/arc_results/test_receipts')
        receipt = dict(payload_sha256='payload', evidence_hashes={'/evidence/result.json': 'hash'},
                       result='/evidence/result.json')
        result = dict(status='done_pending_review', timing_eligible=True, checkpoint_durable=True)
        with patch('pathlib.Path.exists', return_value=True), patch(
                'pathlib.Path.read_text', side_effect=[json.dumps(receipt), json.dumps(result)]), patch(
                'run_arc_multigpu_finish.digest', return_value='hash'):
            self.assertEqual(completed_case({'completion_root': str(root)}, 'payload', 'case'), receipt)
        for changes in ({'timing_eligible': False}, {'checkpoint_durable': False},
                        {'control_only': True}, {'diagnostic_only': True}, {'status': 'failed'}):
            with self.subTest(changes=changes), patch('pathlib.Path.exists', return_value=True), patch(
                    'pathlib.Path.read_text', side_effect=[json.dumps(receipt), json.dumps(dict(result, **changes))]), patch(
                    'run_arc_multigpu_finish.digest', return_value='hash'):
                with self.assertRaisesRegex(ValueError, 'eligible final'):
                    completed_case({'completion_root': str(root)}, 'payload', 'case')
        with patch('pathlib.Path.exists', return_value=True), patch(
                'pathlib.Path.read_text', return_value=json.dumps(receipt)), patch(
                'run_arc_multigpu_finish.digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                completed_case({'completion_root': str(root)}, 'payload', 'case')
        with patch('pathlib.Path.exists', return_value=True), patch(
                'pathlib.Path.read_text', return_value=json.dumps(receipt)):
            with self.assertRaisesRegex(ValueError, 'different frozen payload'):
                completed_case({'completion_root': str(root)}, 'different', 'case')

    def test_source_override_requires_manifest_pin_and_node_local_path(self):
        self.assertEqual(source_campaign({}), Path('/mnt/local/smansou2/paper_multigpu_293571'))
        metadata = dict(source_campaign='/mnt/local/smansou2/new_build', source_manifest_sha256='known')
        with patch('run_arc_multigpu_finish.digest', return_value='known'):
            self.assertEqual(source_campaign(metadata), Path(metadata['source_campaign']))
        with patch('run_arc_multigpu_finish.digest', return_value='different'):
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                source_campaign(metadata)
        for path in ('relative', '/tmp/source', '/mnt/local/smansou2/a/b'):
            with self.assertRaises(ValueError):
                source_campaign(dict(metadata, source_campaign=path))

    def test_final_cohort_is_explicit_and_preserves_engine(self):
        manifest = dict(power_w=300, commit='engine', engine_hashes={'libge2.so': 'hash'})
        result = apply_final_cohort(manifest, dict(power_w=200, cohort='matched_200w_v2'))
        self.assertEqual(result['power_w'], 200)
        self.assertEqual(result['cohort'], 'matched_200w_v2')
        self.assertEqual(result['commit'], 'engine')
        self.assertEqual(result['engine_hashes'], {'libge2.so': 'hash'})
        for power in ('200', 0, 201, True):
            with self.assertRaises(ValueError):
                apply_final_cohort(dict(power_w=300), dict(power_w=power))
        with self.assertRaisesRegex(ValueError, 'Diagnostic'):
            apply_final_cohort(dict(power_w=200, diagnostic_only=True), {})

    def fixture(self, root, graph='fb', system='pipege'):
        old, base, payload = (root/name for name in ('old', 'new', 'payload'))
        for directory in (old/'references', old/'harness/tools', old/'engine', base, payload):
            directory.mkdir(parents=True)
        reference = dict(model=dict(random_seed=17), storage=dict(device_ids=[0],
            prefetch=graph == 'tw', embeddings=dict(options=dict(num_partitions=16, buffer_capacity=4))),
            training=dict(batch_size=50000, negative_sampling=dict(negatives_per_positive=1000)),
            evaluation={})
        (old/'references/single_gpu.yaml').write_text(yaml.safe_dump(reference))
        cfg = multigpu_config(reference, 2, system)
        (old/'references/config.yaml').write_text(yaml.safe_dump(cfg))
        ref_flags = {'GEGE_BOUNDED_COVER_EPOCH_RELABEL': '1'} if system == 'pipege' else {}
        write_json(old/'references/single_gpu.flags.json', ref_flags)
        flags = multigpu_flags(ref_flags) if system == 'pipege' else {}
        if graph == 'tw' and system == 'pipege':
            states = [[0,4,8,12],[0,1,2,3],[0,5,10,15],[0,7,9,14],[0,6,11,13],
                      [1,5,9,13],[4,5,6,7],[1,4,11,14],[1,6,8,15],[1,7,10,12],
                      [2,6,10,14],[8,9,10,11],[2,7,8,13],[2,5,11,12],[2,4,9,15],
                      [3,7,11,15],[12,13,14,15],[3,6,9,12],[3,4,10,13],[3,5,8,14]]
            (old/'references/schedule.txt').write_text('\n'.join(' '.join(map(str, s)) for s in states))
            flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(old/'references/schedule.txt')
        write_json(old/'references/flags.json', flags)
        (old/'harness/tools/eval.py').write_text('# Frozen evaluator\n')
        (old/'ge2.zip').write_bytes(b'original archive')
        (payload/'launcher.py').write_text('# Updated launcher\n')
        spec = dict(config=str(old/'references/config.yaml'),
                    flags=str(old/'references/flags.json'), graph=graph, system=system, gpus=2,
                    data='/canonical/data', model='complex' if graph == 'fb' else 'dot')
        prior = dict(commit='native-source', built_engine_commit='native-build',
                     cases={'case': spec}, engine_hashes={'libge2.so': 'frozen-library'},
                     files={str(p.relative_to(old)): digest(p)
                            for p in (old/'references').iterdir()})
        write_json(old/'manifest.json', prior)
        return old, base, payload, prior

    def test_rebases_references_without_changing_native_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old, base, payload, prior = self.fixture(Path(directory), graph='tw')
            manifest = freeze_retry(old, base, payload, 'case', 'launcher-commit')
            self.assertEqual(manifest['commit'], prior['commit'])
            self.assertEqual(manifest['built_engine_commit'], prior['built_engine_commit'])
            self.assertEqual(manifest['engine_hashes'], prior['engine_hashes'])
            self.assertEqual(manifest['launcher_commit'], 'launcher-commit')
            self.assertEqual(manifest['runtime_policy'], 'default')
            self.assertEqual((base/'engine').resolve(), old/'engine')
            self.assertEqual(manifest['cases']['case']['data'], '/canonical/data')
            flags = json.loads((base/'references/flags.json').read_text())
            self.assertEqual(flags['GEGE_BOUNDED_STATE_ORDER_FILE'], str(base/'references/schedule.txt'))
            self.assertEqual(flags['GEGE_PARTITION_BUFFER_PEER_RELAY'], '1')
            self.assertEqual(json.loads((base/'reference_contract.json').read_text())['batch_per_gpu'], 50000)
            self.assertEqual(json.loads((old/'manifest.json').read_text()), prior)
            self.assertIn('scripts/launcher.py', manifest['files'])
            for rel, expected in manifest['files'].items():
                self.assertEqual(digest(base/rel), expected)

    def test_tw_keeps_original_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            old, base, payload, _ = self.fixture(Path(directory), graph='tw')
            manifest = freeze_retry(old, base, payload, 'case', 'launcher-commit')
            self.assertEqual(manifest['runtime_policy'], 'default')

    def test_fb_transport_for_both_systems(self):
        for system in ('pipege', 'ge2'):
            with self.subTest(system=system), tempfile.TemporaryDirectory() as directory:
                old, base, payload, _ = self.fixture(Path(directory), system=system)
                manifest = freeze_retry(old, base, payload, 'case', 'launcher-commit')
                self.assertEqual(manifest['runtime_policy'], 'c30_nccl_shm_v1')

    def test_restart_paths_preserve_prior_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'multigpu_finish_123'
            base.mkdir()
            (base/'checkpoint').write_text('preserve')
            self.assertEqual(restart_base(base, '0'), base)
            for n in ('1', '2', '10'):
                attempt = restart_base(base, n)
                self.assertEqual(attempt.name, 'multigpu_finish_123_r'+n)
                attempt.mkdir()
            self.assertEqual((base/'checkpoint').read_text(), 'preserve')
            for bad in ('-1', '01', '../x', '', '1.0'):
                with self.assertRaises(ValueError):
                    restart_base(base, bad)

    def test_contract_rejects_training_or_runtime_drift(self):
        for kind in ('batch', 'optimizer', 'peer', 'relabel', 'sampling'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                old, _, _, prior = self.fixture(Path(directory))
                spec = prior['cases']['case']
                self.assertEqual(validate_reference_contract(spec)['status'], 'pass')
                cfg = yaml.safe_load(Path(spec['config']).read_text())
                flags = json.loads(Path(spec['flags']).read_text())
                if kind == 'batch':
                    cfg['training']['batch_size'] = 150000
                elif kind == 'optimizer':
                    cfg['model']['sparse_optimizer'] = dict(type='SGD')
                elif kind == 'peer':
                    flags['GEGE_PARTITION_BUFFER_PEER_RELAY'] = '0'
                elif kind == 'relabel':
                    flags['GEGE_BOUNDED_COVER_EPOCH_RELABEL'] = '0'
                else:
                    cfg['training']['negative_sampling']['superbatch_negative_plan_batches'] = 8
                Path(spec['config']).write_text(yaml.safe_dump(cfg))
                write_json(Path(spec['flags']), flags)
                with self.assertRaises(ValueError):
                    validate_reference_contract(spec)

    def test_reference_itself_cannot_authorize_150k(self):
        with tempfile.TemporaryDirectory() as directory:
            old, _, _, prior = self.fixture(Path(directory))
            for path in (old/'references/single_gpu.yaml', old/'references/config.yaml'):
                cfg = yaml.safe_load(path.read_text())
                cfg['training']['batch_size'] = 150000
                path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                validate_reference_contract(prior['cases']['case'])

    def test_rejects_changed_source_and_experimental_patch(self):
        for kind in ('corrupted', 'patched'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                old, base, payload, prior = self.fixture(Path(directory))
                if kind == 'corrupted':
                    (old/'references/config.yaml').write_text('changed')
                else:
                    prior['ge2_dense_repair'] = {'binary': 'experimental.so'}
                    write_json(old/'manifest.json', prior)
                with self.assertRaises(ValueError):
                    freeze_retry(old, base, payload, 'case', 'launcher-commit')
                self.assertFalse((base/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
