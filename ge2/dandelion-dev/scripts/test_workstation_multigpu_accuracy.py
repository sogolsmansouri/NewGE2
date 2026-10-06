import unittest
from pathlib import Path

from run_workstation_fb_accuracy_control import (check_execution_gate, control_config,
                                                 execution_flags, transport_counts, training_workload,
                                                 epoch_wall_times, EDGES)
from run_workstation_fb_multigpu_queue import CASES


class MultiGpuAccuracyTests(unittest.TestCase):
    def test_single_flags_are_not_changed(self):
        flags = {'GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD': '1'}
        self.assertEqual(flags, execution_flags(flags, 1, 'host'))

    def test_peer_keeps_optimized_math_and_pipeline(self):
        flags = dict(GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1',
                     GEGE_FRAME_CACHE_HIDDEN_FRAMES='6', GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3')
        result = execution_flags(flags, 2, 'peer')
        for key, value in flags.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD'], '1')
        self.assertEqual(result['GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME'], '1')
        self.assertEqual(result['GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK'], '0')
        self.assertEqual(result['GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH'], '0')
        # The native strict checker rejects more than one CUDA buffer.
        self.assertEqual(result['GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET'], '0')

    def test_host_keeps_coordinated_handoffs(self):
        peer = execution_flags({}, 2, 'peer')
        host = execution_flags({}, 2, 'host')
        changed = {key for key in peer if peer[key] != host[key]}
        self.assertEqual(changed, {'GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK'})
        self.assertEqual(host['GEGE_STATEFLOW_PEER_RUNTIME'], 'on')

    def test_independent_scratch_is_explicit(self):
        flags = execution_flags({}, 2, 'peer', 'independent')
        self.assertEqual(flags['GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH'], '1')
        self.assertEqual(flags['GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET'], '0')

    def test_four_gpu_control_requires_its_own_gate(self):
        flags = execution_flags({}, 4, 'peer')
        self.assertEqual(flags['GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD'], '1')
        gate = dict(gpus=2, visible_frames=4, transport='peer', peer_scratch='shared')
        with self.assertRaises(RuntimeError):
            check_execution_gate(gate, 4, 4, 'peer')

    def test_gate_must_match_count_capacity_transport_and_scratch(self):
        gate = dict(gpus=2, visible_frames=4, transport='peer', peer_scratch='shared')
        check_execution_gate(gate, 2, 4, 'peer')
        for changed in ({'gpus': 1}, {'visible_frames': 8}, {'transport': 'host'},
                        {'peer_scratch': 'independent'}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                check_execution_gate(dict(gate, **changed), 2, 4, 'peer')

    def test_config_keeps_batch_per_gpu_q_and_sampler(self):
        template = dict(model=dict(decoder=dict(type='COMPLEX'), encoder=dict(embedding_dim=100)),
                        storage=dict(dataset={}, embeddings=dict(options={})), evaluation={},
                        training=dict(batch_size=50000, dense_sync_batches=1,
                                      negative_sampling=dict(degree_fraction=0.5)))
        config = control_config(template, 'complex', 32, 4, Path('/data'), Path('/model'), epochs=3, gpus=2)
        self.assertEqual(config['storage']['device_ids'], [0, 1])
        self.assertEqual(config['training']['logical_active_devices'], 2)
        self.assertEqual(config['training']['batch_size'], 50000)
        self.assertEqual(config['training']['dense_sync_batches'], 1)
        self.assertEqual(config['training']['negative_sampling'], template['training']['negative_sampling'])
        self.assertEqual(config['storage']['embeddings']['options']['buffer_capacity'], 4)
        with self.assertRaises(ValueError):
            control_config(template, 'complex', 32, 8, Path('/data'), Path('/model'), gpus=2)
        four = control_config(template, 'complex', 32, 4, Path('/data'), Path('/model'), gpus=4)
        self.assertEqual(four['storage']['device_ids'], [0, 1, 2, 3])
        self.assertEqual(four['training']['batch_size'], 50000)

    def test_transport_counters_use_only_epoch_summaries(self):
        text = ('unrelated peer_bytes_executed=900\n'
                '[perf][epoch 1][peer_relay] peer_bytes_executed=123 host_fallback_bytes=0 descriptor_mismatch_count=0\n'
                '[perf][epoch 2][peer_relay] peer_bytes_executed=12 host_fallback_bytes=5 descriptor_mismatch_count=1\n')
        self.assertEqual(transport_counts(text), dict(peer_bytes_executed=135, host_fallback_bytes=5,
                                                     descriptor_mismatch_count=1))

    def test_queue_is_only_complex_one_two_gpu_controls(self):
        self.assertEqual(CASES, (('single', 1, 'host'), ('peer', 2, 'peer'), ('host', 2, 'host')))

    def test_multigpu_workload_does_not_require_single_gpu_progress(self):
        rows = []
        for state in range(88):
            items = EDGES if state == 0 else 0
            rows.append(f'[initializeBatches] device={state % 2} prepare_encode=false task_id=1 items={items} batches=1')
        rows.extend(['Finished training epoch 1', 'Epoch Runtime: 1000ms',
                     '[perf][epoch 1][gpu 0] batches=44', '[perf][epoch 1][gpu 1] batches=44'])
        result = training_workload('\n'.join(rows), 1, 2)
        self.assertEqual(result['epochs'][0]['state_items'], EDGES)
        with self.assertRaises(ValueError):
            training_workload('\n'.join(rows).replace('gpu 1] batches=44', 'gpu 1] batches=43'), 1, 2)
        with self.assertRaises(ValueError):
            training_workload('\n'.join(rows).replace('items='+str(EDGES), 'items=1'), 1, 2)

    def test_wall_times_include_inter_epoch_gap(self):
        text = ('[10/06/26 08:00:00.000] Starting training epoch 1\n'
                '[10/06/26 08:00:10.000] Epoch Runtime: 10000ms\n'
                '[10/06/26 08:00:15.000] Starting training epoch 2\n'
                '[10/06/26 08:00:25.000] Epoch Runtime: 10000ms\n')
        result = epoch_wall_times(text, [10., 10.])
        self.assertEqual(result['epoch_start_to_next_start_s'], [15., 10.])
        self.assertEqual(result['wall_epoch_mean_s'], 12.5)


if __name__ == '__main__':
    unittest.main()
