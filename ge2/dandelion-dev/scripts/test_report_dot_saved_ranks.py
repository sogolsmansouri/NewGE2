import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from report_dot_saved_ranks import derive


class SavedRanks(unittest.TestCase):
    def test_tail_projection_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, ranks = root/'eval.json', root/'ranks.npz'
            np.savez(ranks, head_ranks=np.array([2, 4]), tail_ranks=np.array([1, 2]))
            original = dict(num_eval_edges=2, num_nodes=4, mrr=.5625, hits_at_10=1.,
                            report_directions='both', num_ranks=4,
                            ranks_sha256=hashlib.sha256(ranks.read_bytes()).hexdigest())
            source.write_text(json.dumps(original))
            report = derive(source, ranks, 'tail')
            self.assertEqual(report['mrr'], .75)
            self.assertEqual(report['num_ranks'], 2)
            self.assertFalse(report['derived_report']['rescored'])
            self.assertEqual(json.loads(source.read_text()), original)
            source.write_text(json.dumps(dict(original, mrr=.9)))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                derive(source, ranks, 'tail')
            source.write_text(json.dumps(dict(original, ranks_sha256='bad')))
            with self.assertRaisesRegex(ValueError, 'hash'):
                derive(source, ranks, 'tail')


if __name__ == '__main__':
    unittest.main()
