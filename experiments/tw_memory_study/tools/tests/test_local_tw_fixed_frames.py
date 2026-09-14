import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_local_tw_fixed_frames import cases, pipeline_cases, environment, summarize, graph_memory_preflight, gpu_processes
from queue_local_tw_pipeline_study import predecessor_finished, study_rows


class FixedFramesTest(unittest.TestCase):
    def test_selected_gpu_monitor_does_not_claim_other_users_gpu(self):
        with patch.dict('os.environ', {'TW_PHYSICAL_GPU': 'GPU-mine'}):
            with patch('run_local_tw_fixed_frames.subprocess.check_output', side_effect=[
                    'GPU-mine\n', 'GPU-other, 123\nGPU-mine, 456\n']):
                self.assertEqual(gpu_processes(), [456])

    def test_visible_hidden_has_constant_partition_and_frame_memory(self):
        rows = study_rows('visible-hidden')
        self.assertEqual(len(rows), 7)
        self.assertEqual({r['q'] for r in rows}, {3, 4, 5, 6, 7})
        self.assertEqual({r['p'] for r in rows}, {16})
        self.assertEqual({r['k'] for r in rows}, {7})
        self.assertEqual(len({r['frame_bytes'] * r['k'] for r in rows}), 1)
        self.assertTrue(all(r['q'] + r['hp'] + r['hs'] == 7 for r in rows))
        with self.assertRaises(ValueError):
            cases(q=7, min_frames=7, max_frames=7, partitions=6)
        with self.assertRaises(ValueError):
            cases(q=4, min_frames=7, max_frames=7, partitions=5)

    def test_queue_waits_for_service_not_gpu_gaps(self):
        for state in ('active', 'activating', 'deactivating', 'reloading'):
            self.assertFalse(predecessor_finished({'LoadState': 'loaded', 'ActiveState': state}, {'status': 'finished'}))
        self.assertTrue(predecessor_finished({'LoadState': 'loaded', 'ActiveState': 'inactive'}, {}))
        self.assertTrue(predecessor_finished({'LoadState': 'loaded', 'ActiveState': 'failed'}, {}))
        self.assertFalse(predecessor_finished({}, {}))
        self.assertFalse(predecessor_finished({'LoadState': 'not-found'}, {'status': 'training'}))
        self.assertTrue(predecessor_finished({'LoadState': 'not-found'}, {'status': 'finished'}))

    def test_pipeline_controls_retain_frames_but_disable_payload_preload(self):
        rows = pipeline_cases()
        self.assertEqual(len(rows), 4)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'artifacts').mkdir()
            (run / 'artifacts/reference_flags.sh').write_text('')
            for row in rows:
                row['max_admits'] = 3
                env = environment(run, run, row, 1024)
                self.assertEqual(env['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '3')
                self.assertEqual(env['GEGE_FRAME_CACHE_AUTO_PIPELINE_FRAMES'], '0')
                if row['parameter_pipeline']:
                    self.assertEqual(env['GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES'], '-1')
                    self.assertEqual(env['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'], '1')
                else:
                    self.assertEqual(env['GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES'], '0')
                    self.assertEqual(env['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'], '0')

    def test_pipeline_summary_requires_actual_behavior(self):
        for row in pipeline_cases():
            row.update(states=20, overlap=19)
            param = str(row['parameter_pipeline']).lower()
            graph = str(row['graph_prefetch']).lower()
            prefix = ('Epoch Runtime: 1000ms\n' * 5 +
                      'Ordering states=20 transitions=19 total_buckets=256 max_admits=3 transition_admits=57\n')
            alloc = ('[startup-timing][MemPartitionBuffer::ctor] deferred backing allocation device=cuda:0 visible_rows=256 physical_rows=448 dim=100 pinned=true hidden_frames=3\n' * 2
                     if row['shared_hidden'] else
                     '[fixed-frame-budget] visible=4 preload=0 stale=3 physical=7 partition_rows=64 width=100 bytes=179200\n' * 2)
            events = (f'[outer-update 1] prefetch={graph}\npreloaded_admit={param} '
                      f'hidden_publish_parts={3 if row["parameter_pipeline"] else 0} '
                      f'deferred_stale_writeback={param}\n')
            self.assertEqual(summarize(prefix + alloc + events, row, 5, 0, set())['status'], 'valid_timing')
            wrong = events.replace(f'prefetch={graph}', f'prefetch={str(not row["graph_prefetch"]).lower()}')
            self.assertFalse(summarize(prefix + alloc + wrong, row, 5, 0, set())['control_ok'])

    def test_q3_rows_and_reference(self):
        rows = cases(q=3, baseline_p=7)
        self.assertEqual(len(rows), 13)
        self.assertEqual(sorted({r['p'] for r in rows}), [5, 7, 8, 10, 11, 13, 14, 16])
        for r in rows:
            self.assertEqual(3 + r['hp'] + r['hs'], r['k'])
            self.assertLessEqual(r['k'] * r['frame_bytes'], 20 * 2**30)
        self.assertEqual({(r['hp'], r['hs']) for r in rows if r['k'] == 6}, {(1, 2), (2, 1)})
        self.assertTrue(any(r['k'] == 3 and r['p'] == 7 for r in rows))

    def test_invalid_geometry(self):
        for kwargs in (dict(q=1), dict(q=3, min_frames=2), dict(frame_budget_gib=0), dict(q=3, baseline_p=2)):
            with self.assertRaises(ValueError):
                cases(**kwargs)

    def test_shared_pool_counts_each_physical_frame_once(self):
        rows = cases(frame_policy='shared')
        self.assertEqual(len(rows), 7)
        self.assertEqual([r['p'] for r in rows], [7, 8, 10, 11, 13, 14, 16])
        for row in rows[1:]:
            self.assertTrue(row['shared_hidden'])
            self.assertEqual(row['hp'], row['k'] - row['q'])
            self.assertEqual(row['hs'], row['k'] - row['q'])
        with self.assertRaises(ValueError):
            cases(frame_policy='bad')

    def test_off_cannot_pass_with_consumed_hidden_admits(self):
        row = pipeline_cases()[0]
        row.update(states=20, overlap=19)
        text = ('Epoch Runtime: 1000ms\n' * 5 +
                'Ordering states=20 transitions=19 total_buckets=256 max_admits=3 transition_admits=57\n' +
                '[fixed-frame-budget] visible=4 preload=0 stale=3 physical=7 partition_rows=64 width=100 bytes=179200\n' * 2 +
                '[outer-update 1] prefetch=false\npreloaded_admit=true hidden_publish_parts=3 deferred_stale_writeback=false\n')
        self.assertFalse(summarize(text, row, 5, 0, set())['control_ok'])

    def test_shared_partial_fallback_is_not_an_extra_gpu_stage(self):
        row = cases(q=4, min_frames=5, max_frames=5, partitions=16, frame_policy='shared')[0]
        row.update(states=20, overlap=19)
        text = ('Epoch Runtime: 1000ms\n' * 5 +
                'Ordering states=20 transitions=19 total_buckets=256 max_admits=3 transition_admits=57\n' +
                '[startup-timing][MemPartitionBuffer::ctor] deferred backing allocation device=cuda:0 visible_rows=256 physical_rows=320 dim=100 pinned=true hidden_frames=1\n' * 2 +
                '[outer-update 1] prefetch=true\npreloaded_admit=true hidden_publish_parts=1 deferred_stale_writeback=true\n' +
                '[partition-buffer-preload-consume] rows=0 visible_install_parts=0 hidden_publish_parts=1\n' +
                '[partition-buffer-swap] fallback_visible_admit_parts=2\n')
        self.assertTrue(summarize(text, row, 5, 0, set())['control_ok'])
        self.assertFalse(summarize(text.replace('rows=0', 'rows=128'), row, 5, 0, set())['control_ok'])

    def test_q3_summary_uses_actual_capacity(self):
        row = dict(q=3, k=5, p=7, hp=1, hs=1, states=7, overlap=6)
        text = ('Epoch Runtime: 1000ms\n' * 5 +
                'Ordering states=7 transitions=6 total_buckets=49 max_admits=2 transition_admits=12\n' +
                '[fixed-frame-budget] visible=3 preload=1 stale=1 physical=5 partition_rows=5 width=8 bytes=800\n' * 2)
        self.assertEqual(summarize(text, row, 5, 0, set())['status'], 'valid_timing')
        self.assertFalse(summarize(text.replace('visible=3', 'visible=4'), row, 5, 0, set())['allocation_ok'])
        short = text.replace('Epoch Runtime: 1000ms\n' * 5, 'Epoch Runtime: 1000ms\n' * 3)
        self.assertEqual(summarize(short, row, 3, 0, set())['status'], 'valid_timing')
        self.assertEqual(summarize(short, row, 5, 0, set())['status'], 'failed_or_invalid')

    def test_memory_preflight_counts_full_graph_not_only_assigned_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            (root / 'edges').mkdir()
            (root / 'artifacts/p4.txt').write_text('state=[0,1,2]\nstate=[1,2,3]\n')
            (root / 'edges/train_partition_offsets.txt').write_text('1\n' * 16)
            row = dict(p=4, k=3, q=3, frame_bytes=100)
            result = graph_memory_preflight(root, row, root, 500)
            self.assertEqual(result['state_graph_bytes'], [144, 144])
            self.assertEqual(result['minimum_peak_payload_bytes'], 588)
            self.assertEqual(result['status'], 'infeasible_payload')
            self.assertEqual(graph_memory_preflight(root, row, root, 600)['status'], 'requires_runtime_memory_gate')
            row['graph_prefetch'] = False
            self.assertEqual(graph_memory_preflight(root, row, root, 500)['minimum_peak_payload_bytes'], 444)

    def test_memory_rows_and_odd_variants(self):
        rows = cases()
        self.assertEqual(len(rows), 10)
        self.assertEqual(sorted({r['p'] for r in rows}), [7, 8, 10, 11, 13, 14, 16])
        for r in rows:
            self.assertEqual(4 + r['hp'] + r['hs'], r['k'])
            self.assertLessEqual(r['k'] * r['frame_bytes'], 20 * 2**30)
        self.assertEqual({(r['hp'], r['hs']) for r in rows if r['k'] == 5}, {(0, 1), (1, 0)})

    def test_no_success_from_exit_status_alone(self):
        row = dict(k=5, p=8, hp=1, hs=0, states=6, overlap=10)
        report = summarize('Epoch Runtime: 1000ms\n' * 5, row, 5, 0, set())
        self.assertEqual(report['status'], 'failed_or_invalid')

    def test_extra_frames_fail_even_with_complete_training(self):
        row = dict(k=5, p=8, hp=1, hs=0, states=6, overlap=10)
        text = ('Epoch Runtime: 1000ms\n' * 5 +
                'Ordering states=6 transitions=5 total_buckets=64 max_admits=2 transition_admits=10\n' +
                '[fixed-frame-budget] visible=4 preload=1 stale=0 physical=6 partition_rows=5 width=8 bytes=960\n' * 2)
        self.assertFalse(summarize(text, row, 5, 0, set())['allocation_ok'])
        self.assertEqual(summarize(text.replace('physical=6', 'physical=5'), row, 5, 0, set())['status'], 'valid_timing')


if __name__ == '__main__':
    unittest.main()
