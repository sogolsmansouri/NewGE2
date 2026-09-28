import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from run_arc_multigpu_campaign import check_replicas, multigpu_config, multigpu_flags, split_hash_for_view, training_check
from prepare_tw_multigpu import flags_for


class MultiGpuCampaignTests(unittest.TestCase):
    def test_repartitioned_heldout_rows_use_view_hash(self):
        canonical = dict(splits=dict(validation=dict(sha256='source')))
        view = dict(num_partitions=32, splits=dict(validation=dict(source_sha256='source', output_sha256='reordered')))
        self.assertEqual(split_hash_for_view('validation', canonical, view, 32), 'reordered')
        self.assertEqual(split_hash_for_view('validation', canonical, {}, 16), 'source')
        with self.assertRaises(ValueError):
            split_hash_for_view('validation', canonical, view, 16)
        view['splits']['validation']['source_sha256'] = 'different'
        with self.assertRaises(ValueError):
            split_hash_for_view('validation', canonical, view, 32)

    def test_preserves_fb_reference_and_disables_native_eval(self):
        reference = dict(model=dict(random_seed=17), storage=dict(device_ids=[0], prefetch=False,
            embeddings=dict(options=dict(num_partitions=32, buffer_capacity=4))),
            training=dict(batch_size=50000, dense_sync_batches=1, num_epochs=10,
                negative_sampling=dict(num_chunks=50, negatives_per_positive=1000, degree_fraction=.5)),
            evaluation=dict(epochs_per_eval=1000))
        original = copy.deepcopy(reference)
        for count in (2, 4):
            multi = multigpu_config(reference, count, 'pipege')
            self.assertEqual(multi['model'], original['model'])
            self.assertEqual(multi['training']['negative_sampling'], original['training']['negative_sampling'])
            self.assertEqual(multi['training']['batch_size'], 50000)
            self.assertEqual(multi['training']['dense_sync_batches'], 1)
            self.assertEqual(multi['training']['logical_active_devices'], count)
            self.assertFalse(multi['storage']['prefetch'])
        self.assertEqual(reference, original)
        reference['training']['batch_size'] = 150000
        with self.assertRaises(ValueError):
            multigpu_config(reference, 4, 'pipege')

    def test_flags_preserve_relabel_and_frame_budget(self):
        flags = multigpu_flags(dict(GEGE_BOUNDED_COVER_EPOCH_RELABEL='1', GEGE_BOUNDED_COVER_RELABEL_SEED='17',
                              GEGE_FRAME_CACHE_HIDDEN_FRAMES='6', GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES='-1'))
        self.assertEqual(flags['GEGE_BOUNDED_COVER_EPOCH_RELABEL'], '1')
        self.assertEqual(flags['GEGE_BOUNDED_COVER_RELABEL_SEED'], '17')
        self.assertEqual(flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '6')
        self.assertEqual(flags['GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES'], '-1')
        self.assertEqual(flags['GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD'], '1')
        self.assertEqual(flags['GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET'], '0')
        self.assertEqual(flags['GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH'], '1')
        with self.assertRaises(ValueError):
            multigpu_flags({'GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES': '3'})

    def log(self, spec, count):
        rows = (spec['nodes']+spec['p']-1)//spec['p']
        lines = [f'Broadcasting model to: {count} GPUs', 'SynchronousMultiGPUTrainer',
            f'Stateflow multi-GPU selected family=CUSTOM gpu_count={count} lanes={count} microstates={spec["states"]}']
        for gpu in range(count):
            lines += [f'deferred backing allocation device=cuda:{gpu} visible_rows={4*rows} physical_rows={(4+spec["hidden"])*rows} dim=100 pinned=true hidden_frames={spec["hidden"]}']*2
        for epoch in range(2):
            lines += [f'[bounded-cover-relabel] epoch={epoch} seed=17',
                f'Edges processed: [{spec["edges"]}/{spec["edges"]}], 100.00%',
                f'Finished training epoch {epoch+1}', 'Epoch Runtime: 100000ms']
        lines += [f'[stateflow-peer-validate {i}] dst_mismatch_values=0 src_mismatch_values=0' for i in range(16)]
        return '\n'.join(lines)

    def test_gate_requires_edges_frames_peer_checks_and_relabel(self):
        helper = types.ModuleType('run_arc_pipege_quality')
        helper.timing_summary = lambda text, times: dict(epoch_times_s=times)
        with patch.dict(sys.modules, run_arc_pipege_quality=helper):
            for graph, p, hidden, states in [('fb', 32, 6, 88), ('tw', 16, 3, 20)]:
                spec = dict(system='pipege', graph=graph, p=p, hidden=hidden, states=states,
                            width=100, nodes=86054151, edges=304727650)
                for count in (2, 4):
                    log = self.log(spec, count)
                    self.assertEqual(training_check(log, spec, count, 2, True)['epoch_times_s'], [100, 100])
                    for broken in (log.replace('100.00%', '99.99%'), log.replace('hidden_frames=', 'hidden='),
                                   log.replace('dst_mismatch_values=0', 'dst_mismatch_values=1'),
                                   log.replace('[stateflow-peer-validate', '[disabled'),
                                   log+'\nCUDA error'):
                        with self.assertRaises(ValueError):
                            training_check(broken, spec, count, 2, True)
                    if graph == 'fb':
                        with self.assertRaises(ValueError):
                            training_check(log.replace('epoch=1 seed=17', 'epoch=0 seed=17'), spec, count, 2, True)

    def test_dense_replica_equality(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.jit.trace(torch.nn.Linear(3, 2), torch.zeros(1, 3))
            for i in range(2):
                torch.jit.save(model, str(root/f'model.pt_{i}'))
            self.assertTrue(check_replicas(root, 2)['equal'])
            with torch.no_grad():
                model.weight.add_(1)
            torch.jit.save(model, str(root/'model.pt_1'))
            with self.assertRaises(ValueError):
                check_replicas(root, 2)


if __name__ == '__main__':
    unittest.main()
