import json
from pathlib import Path
import tempfile
import unittest

from arc_job_support import write_json
from run_arc_multigpu_finish import freeze_retry
from run_arc_paper_case import digest


class MultiGpuFinishTests(unittest.TestCase):
    def fixture(self, root, graph='fb'):
        old, base, payload = (root/name for name in ('old', 'new', 'payload'))
        for directory in (old/'references', old/'harness/tools', old/'engine', base, payload):
            directory.mkdir(parents=True)
        (old/'references/config.yaml').write_text('training: {}\n')
        (old/'references/order.txt').write_text('0 1 2 3\n')
        write_json(old/'references/flags.json', {
            'GEGE_BOUNDED_STATE_ORDER_FILE': str(old/'references/order.txt'),
            'GEGE_STATEFLOW_PEER_RELAY': '1'})
        (old/'harness/tools/eval.py').write_text('# Frozen evaluator\n')
        (old/'ge2.zip').write_bytes(b'original archive')
        (payload/'launcher.py').write_text('# Updated launcher\n')
        spec = dict(config=str(old/'references/config.yaml'),
                    flags=str(old/'references/flags.json'), graph=graph,
                    data='/canonical/data', model='complex' if graph == 'fb' else 'dot')
        prior = dict(commit='native-source', built_engine_commit='native-build',
                     cases={'case': spec}, engine_hashes={'libge2.so': 'frozen-library'},
                     files={str(p.relative_to(old)): digest(p)
                            for p in (old/'references').iterdir()})
        write_json(old/'manifest.json', prior)
        return old, base, payload, prior

    def test_rebases_references_without_changing_native_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old, base, payload, prior = self.fixture(Path(directory))
            manifest = freeze_retry(old, base, payload, 'case', 'launcher-commit')
            self.assertEqual(manifest['commit'], prior['commit'])
            self.assertEqual(manifest['built_engine_commit'], prior['built_engine_commit'])
            self.assertEqual(manifest['engine_hashes'], prior['engine_hashes'])
            self.assertEqual(manifest['launcher_commit'], 'launcher-commit')
            self.assertEqual(manifest['runtime_policy'], 'c30_nccl_shm_v1')
            self.assertEqual((base/'engine').resolve(), old/'engine')
            self.assertEqual(manifest['cases']['case']['data'], '/canonical/data')
            flags = json.loads((base/'references/flags.json').read_text())
            self.assertEqual(flags['GEGE_BOUNDED_STATE_ORDER_FILE'], str(base/'references/order.txt'))
            self.assertEqual(flags['GEGE_STATEFLOW_PEER_RELAY'], '1')
            self.assertEqual(json.loads((old/'manifest.json').read_text()), prior)
            self.assertIn('scripts/launcher.py', manifest['files'])
            for rel, expected in manifest['files'].items():
                self.assertEqual(digest(base/rel), expected)

    def test_tw_keeps_original_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            old, base, payload, _ = self.fixture(Path(directory), graph='tw')
            manifest = freeze_retry(old, base, payload, 'case', 'launcher-commit')
            self.assertEqual(manifest['runtime_policy'], 'default')

    def test_rejects_changed_source_and_experimental_patch(self):
        for kind in ('corrupted', 'patched'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                old, base, payload, prior = self.fixture(Path(directory))
                if kind == 'corrupted':
                    (old/'references/order.txt').write_text('changed')
                else:
                    prior['ge2_dense_repair'] = {'binary': 'experimental.so'}
                    write_json(old/'manifest.json', prior)
                with self.assertRaises(ValueError):
                    freeze_retry(old, base, payload, 'case', 'launcher-commit')
                self.assertFalse((base/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
