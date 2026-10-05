import unittest

import numpy as np
import torch

from diagnose_partition_ranking import (
    DotRanker, configure_scoring, partition_counts, same_partition_mask, validate_reference,
)


class PartitionRankingTest(unittest.TestCase):
    def setUp(self):
        configure_scoring()

    def test_uneven_final_partition_and_chunk_crossing(self):
        tails = torch.tensor([0, 5, 16])
        mask = same_partition_mask(tails, 3, 8, 17, 4)
        self.assertEqual(mask.tolist(), [[True, True, False, False, False],
                                        [False, False, True, True, True],
                                        [False, False, False, False, False]])

    def test_invalid_counts_and_intervals(self):
        for nodes, parts, start, stop in [(0, 1, 0, 1), (8, 0, 0, 1), (8, 9, 0, 1),
                                         (8, 2, -1, 2), (8, 2, 4, 4), (8, 2, 0, 9)]:
            with self.assertRaises(ValueError):
                same_partition_mask(torch.tensor([0]), start, stop, nodes, parts)

    def test_streamed_counts_match_brute_force_with_filtered_ties(self):
        candidates = torch.tensor([[1., 0.], [1., 0.], [2., 0.], [0., 0.],
                                   [0., 2.], [0., 1.], [0., 1.], [1., 1.]])
        queries = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        tails = torch.tensor([0, 5, 7])
        targets = candidates[tails]
        scores = queries @ candidates.T
        scores[torch.arange(3), tails] = -torch.inf
        scores[0, 2] = -torch.inf
        ranker = DotRanker(queries, targets, 'pessimistic')
        for p in (1, 2, 4):
            total = torch.zeros(3, dtype=torch.int64)
            for lo, hi in [(0, 3), (3, 6), (6, 8)]:
                total += partition_counts(scores[:, lo:hi], candidates[lo:hi], ranker,
                                          tails, lo, 8, p, 0, candidates.abs().max().double())
            mask = same_partition_mask(tails, 0, 8, 8, p)
            expected = ((scores.double() >= ranker.positive[:, None]) & mask).sum(1)
            self.assertTrue(torch.equal(total, expected))

    def test_reference_gate(self):
        triples = np.array([[0, 0, 1], [1, 0, 2]])
        ranks = np.array([1, 20])
        reference = dict(report_directions='tail', filtered=True, tf32=False, mapping_contract='identity',
                         ranking_contract='fp32_query_fp64_dot_refinement_v1', num_eval_edges=2,
                         num_nodes=32, mrr=.525, hits_at_10=.5)
        validate_reference(reference, triples, ranks)
        for key, value in [('filtered', False), ('tf32', True), ('mrr', .6),
                           ('report_directions', 'both'), ('mapping_contract', 'remapped')]:
            with self.assertRaises(ValueError):
                validate_reference(dict(reference, **{key: value}), triples, ranks)


if __name__ == '__main__':
    unittest.main()
