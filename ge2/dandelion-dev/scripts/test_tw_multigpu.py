#!/usr/bin/env python3
import copy
import unittest

from prepare_tw_multigpu import check_config, cover_check, flags_for, make_pipege
from run_tw_multigpu import check_evaluation, check_resource_policy, parse_training, qualify_timing


class PreparationTests(unittest.TestCase):
    def test_shared_mode_never_allows_selected_gpu_contention(self):
        for shared in (False, True):
            with self.assertRaises(RuntimeError):
                check_resource_policy([12], '12', [], shared)

    def test_shared_mode_is_opt_in(self):
        with self.assertRaises(RuntimeError):
            check_resource_policy([], '99', ['other_job'], False)
        check_resource_policy([], '99', ['other_job'], True)
        check_resource_policy([], '', [], False)

    def test_shared_timing_is_always_provisional(self):
        clean = [dict(other_jobs=[], other_processes=[])]
        self.assertEqual(qualify_timing(clean, ['GPU-0'], True), (True, 'shared_node_provisional'))
        busy = [dict(other_jobs=['other_job'], other_processes=['GPU-3, 99, python'])]
        self.assertEqual(qualify_timing(busy, ['GPU-0', 'GPU-1'], True), (False, 'shared_node_provisional'))
        with self.assertRaises(RuntimeError):
            qualify_timing(busy, ['GPU-0'], False)
        with self.assertRaises(RuntimeError):
            qualify_timing(busy, ['GPU-3'], True)
        with self.assertRaises(RuntimeError):
            qualify_timing([], ['GPU-0'], True)

    def config(self):
        return dict(model=dict(encoder=dict(layers=[[dict(output_dim=100)]]),
                    dense_optimizer=dict(type='ADAGRAD', options=dict(learning_rate=.1)),
                    sparse_optimizer=dict(type='ADAGRAD', options=dict(learning_rate=.1))),
            storage=dict(device_ids=[0], prefetch=True,
                         embeddings=dict(options=dict(num_partitions=16, buffer_capacity=4))),
            training=dict(batch_size=50000, num_epochs=10, negative_sampling_method='RNS',
                          negative_sampling=dict(num_chunks=50, negatives_per_positive=1000, degree_fraction=.5)),
            evaluation={})

    def test_gpu_count_not_global_batch(self):
        ref = self.config()
        for gpus in (2, 4):
            cfg = make_pipege(ref, gpus)
            check_config(cfg, 'pipege', gpus)
            self.assertEqual(cfg['training']['batch_size'], 50000)
            self.assertEqual(cfg['training']['logical_active_devices'], gpus)
        self.assertEqual(ref['storage']['device_ids'], [0])

    def test_reject_150k(self):
        cfg = self.config()
        cfg['training']['batch_size'] = 150000
        with self.assertRaises(ValueError):
            make_pipege(cfg, 2)

    def test_peer_scope_and_loss(self):
        flags = flags_for({})
        self.assertEqual(flags['GEGE_STATEFLOW_PEER_RUNTIME_SCOPE'], 'all')
        self.assertEqual(flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'], '1')
        with self.assertRaises(ValueError):
            flags_for({'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE': '8'})

    def test_bad_cover(self):
        with self.assertRaises(ValueError):
            cover_check('0 1 2 3\n'*20)

    def test_valid_affine_cover_both_encodings(self):
        states = [[0,4,8,12],[0,1,2,3],[0,5,10,15],[0,7,9,14],[0,6,11,13],
                  [1,5,9,13],[4,5,6,7],[1,4,11,14],[1,6,8,15],[1,7,10,12],
                  [2,6,10,14],[8,9,10,11],[2,7,8,13],[2,5,11,12],[2,4,9,15],
                  [3,7,11,15],[12,13,14,15],[3,6,9,12],[3,4,10,13],[3,5,8,14]]
        for text in ('\n'.join(' '.join(map(str, s)) for s in states),
                     '\n'.join('state='+str(s) for s in states)):
            report = cover_check(text)
            self.assertEqual(report['unique_pairs'], 120)
            self.assertEqual(len(report['disjoint_parallel_classes']), 5)

    def log(self, gpus=2, checks=16):
        rows = (41652230+15)//16
        lines = [f'Broadcasting model to: {gpus} GPUs', 'SynchronousMultiGPUTrainer',
                 f'Stateflow multi-GPU selected rounds={20//gpus} active_devices={gpus}',
                 f'Stateflow multi-GPU selected family=CUSTOM:lane_matched gpu_count={gpus} buffer_capacity=4 lanes={gpus} microstates=20 estimated_bucket_edges:1321528663',
                 '[manual_dot_rns] enabled=1', 'Using bucket-streaming LP path', 'stateflow_scope=all']
        for gpu in range(gpus):
            lines.extend([f'deferred backing allocation device=cuda:{gpu} visible_rows={4*rows} physical_rows={7*rows} dim=100 pinned=true hidden_frames=3']*2)
        lines.extend(f'[stateflow-peer-validate {i}] dst_mismatch_values=0 src_mismatch_values=0' for i in range(checks))
        lines.extend(['Finished training epoch 1', 'Epoch Runtime: 100000ms',
                      'Finished training epoch 2', 'Epoch Runtime: 90000ms'])
        return '\n'.join(lines)

    def test_complete_gate(self):
        for gpus in (2, 4):
            result = parse_training(self.log(gpus), 'pipege', gpus, 2, True)
            self.assertEqual(result['average_epoch_s'], 95)
            self.assertEqual(result['steady_average_epoch_s_excluding_first'], 90)

    def test_fail_closed_gates(self):
        for text in (self.log(checks=0), self.log().replace('dst_mismatch_values=0', 'dst_mismatch_values=1'),
                     self.log().replace('microstates=20', 'microstates=35'),
                     self.log().replace('1321528663', '1468345182'),
                     self.log().replace('hidden_frames=3', 'hidden_frames=6'),
                     self.log().replace('Finished training epoch 2', 'Finished training epoch 1')):
            with self.assertRaises(ValueError):
                parse_training(text, 'pipege', 2, 2, True)

    def test_ge2_no_pipege_markers(self):
        text = 'Broadcasting model to: 4 GPUs\nSynchronousMultiGPUTrainer\nFinished training epoch 1\nEpoch Runtime: 100ms\nFinished training epoch 2\nEpoch Runtime: 90ms'
        self.assertEqual(len(parse_training(text, 'ge2', 4, 2)['epoch_times_s']), 2)

    def test_eval_direction(self):
        metrics = dict(eval_edges_sha256='abc', num_ranks=10000, report_directions='tail',
                       filtered=True, tf32=False, tie_policy='pessimistic', mrr=.015, hits_at_10=.034)
        check_evaluation(metrics, 'abc')
        for change in (dict(num_ranks=20000), dict(report_directions='both'), dict(tf32=True),
                       dict(mrr=float('nan')), dict(filtered=False)):
            bad = copy.deepcopy(metrics)
            bad.update(change)
            with self.assertRaises(ValueError):
                check_evaluation(bad, 'abc')


if __name__ == '__main__':
    unittest.main()
