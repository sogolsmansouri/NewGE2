import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_arc_paper_case import archive_checkpoint, checkpoint_entries, hardware_check, digest, verify_evaluation_artifacts


class PaperCaseTests(unittest.TestCase):
    def test_power_and_isolation(self):
        row = dict(gpu='0, GPU-test, 200.00 W, 60 W', other_jobs=[], other_processes=[])
        hardware_check([row], 200, 'GPU-test')
        for changed in (dict(row, other_jobs=['42']), dict(row, other_processes=['pid']),
                        dict(row, gpu='0, GPU-test, 300 W'), dict(row, gpu='0, GPU-test, nan W'), dict(row, gpu='')):
            with self.assertRaises(ValueError):
                hardware_check([changed], 200, 'GPU-test')
        with self.assertRaises(ValueError):
            hardware_check([], 200, 'GPU-test')

    def test_unsafe_checkpoint_paths(self):
        for path in ('../outside', '/outside', 'nested/file'):
            with self.assertRaises(ValueError):
                checkpoint_entries(dict(files=[dict(path=path)]))
        with self.assertRaises(ValueError):
            checkpoint_entries(dict(files=[]))

    def test_evaluation_identity_and_metric_recalculation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root/'embeddings.bin'
            weights.write_bytes(b'weights')
            ranks = root/'eval.ranks.npz'
            np.savez(ranks, tail_ranks=np.full(10000, 2, dtype=np.int64))
            checkpoint = dict(source=str(root), files=[dict(path=weights.name, sha256=digest(weights))])
            quality = dict(embedding_file=str(weights), ranks_file=str(ranks), ranks_sha256=digest(ranks),
                           mrr=.5, hits_at_10=1.)
            self.assertTrue(verify_evaluation_artifacts(quality, checkpoint)['recomputed_tail_metrics'])
            verify_evaluation_artifacts(dict(quality, entity_bin_sha256=digest(weights)), checkpoint)
            for changed in (dict(quality,mrr=.6), dict(quality,entity_bin_sha256='bad'),
                            dict(quality,embedding_file=str(root/'wrong')),dict(quality,ranks_sha256='bad')):
                with self.assertRaises(ValueError):
                    verify_evaluation_artifacts(changed, checkpoint)

    def test_archive_is_verified_and_source_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.mkdir()
            (source/'model.pt').write_bytes(b'weights')
            manifest = root/'checkpoint.json'
            manifest.write_text(json.dumps(dict(source=str(source), files=[dict(path='model.pt', bytes=7,
                sha256=hashlib.sha256(b'weights').hexdigest())])))
            result = archive_checkpoint(manifest, root/'archive')
            self.assertEqual(result['status'], 'verified')
            self.assertTrue((source/'model.pt').exists())
            self.assertEqual((root/'archive/model.pt').read_bytes(), b'weights')
            (root/'archive/model.pt').write_bytes(b'corrupt')
            archive_checkpoint(manifest, root/'archive')
            self.assertEqual((root/'archive/model.pt').read_bytes(), b'weights')
            (source/'model.pt').write_bytes(b'changed')
            (root/'archive/model.pt').unlink()
            with self.assertRaises(ValueError):
                archive_checkpoint(manifest, root/'archive')


if __name__ == '__main__':
    unittest.main()
