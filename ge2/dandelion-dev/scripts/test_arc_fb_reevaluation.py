from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_arc_fb_reevaluation import command_for, digest, validate_protocol, verify_result


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(report_directions='tail', filtered=True, tie_policy='pessimistic',
            tf32=False, mapping_contract='identity', num_eval_edges=10000, num_ranks=10000,
            num_nodes=86054151, num_relations=14824, entity_shape=[86054151, 100],
            score='complex', score_contract='ge2_forward_inverse_relation_embeddings',
            ranking_contract='fp32_query_fp64_dot_refinement_v1', sample_seed=None,
            batch_size=128, candidate_chunk=250000, filter_chunk=5000000,
            entity_bin_sha256='entity', src_relation_bin_sha256='src',
            dst_relation_bin_sha256='dst', eval_edges_sha256='queries')

    def test_command_explicit_tail_and_fixed_panel(self):
        paths = {key: '/fixture/'+key for key in ('entity_bin', 'src_relation_bin',
            'dst_relation_bin', 'ge2_data_dir', 'eval_edges')}
        command = command_for(self.old, paths, Path('/tools'), Path('/out.json'), 64)
        self.assertEqual(command[command.index('--report-directions')+1], 'tail')
        self.assertEqual(command[command.index('--num-test')+1], '64')
        self.assertIn('--filtered', command)
        self.assertNotIn('--sample-seed', command)

    def test_reject_mismatched_protocol(self):
        for key, value in dict(report_directions='both', filtered=False, tf32=True,
                              num_nodes=100, sample_seed=42, entity_shape=[86054151, 50]).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(dict(self.old, **{key: value}))

    def make_ranks(self, folder, ranks=(1, 10), triples=None):
        previous = folder/'previous.npz'
        current = folder/'current.npz'
        pairs = np.array([[1, 2, 3], [3, 2, 4]], dtype=np.int64)
        np.savez(previous, triples=pairs, tail_ranks=np.array([1, 10], dtype=np.int64))
        ranks = np.asarray(ranks)
        np.savez(current, triples=pairs if triples is None else triples, tail_ranks=ranks)
        old = dict(self.old, ranks_sha256=digest(previous))
        result = dict(old, num_ranks=2, num_eval_edges=2, ranks_file=str(current),
                      ranks_sha256=digest(current), mrr=float(np.mean(1.0/ranks)),
                      hits_at_10=float(np.mean(ranks <= 10)))
        return old, result, previous

    def test_metrics_and_rank_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            old, result, previous = self.make_ranks(Path(directory), (1, 11))
            comparison = verify_result(old, result, previous, 2)
            self.assertEqual(comparison['changed_ranks'], 1)
            self.assertEqual(comparison['hits_at_10'], 0.5)

    def test_reject_wrong_query_order(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_ranks(Path(directory), triples=np.array([[3, 2, 4], [1, 2, 3]]))
            with self.assertRaisesRegex(ValueError, 'Query identity'):
                verify_result(*args, 2)

    def test_reject_corrupt_metrics_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old, result, previous = self.make_ranks(Path(directory))
            for key, value in dict(mrr=0.8, entity_bin_sha256='wrong', num_ranks=4).items():
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_result(old, dict(result, **{key: value}), previous, 2)

    def test_reject_invalid_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            for ranks in ((-1, 2), (float('nan'), 2), (1, 86054152)):
                with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                    verify_result(*self.make_ranks(Path(directory), ranks), 2)


if __name__ == '__main__':
    unittest.main()
