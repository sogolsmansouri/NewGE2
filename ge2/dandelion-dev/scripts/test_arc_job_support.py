import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from arc_job_support import AllocationDeadline, probe, run_logged, save_failure_evidence
from run_arc_paper_case import hardware_check
from submit_arc_paper_campaign import selected_cells


class Evidence(unittest.TestCase):
    def test_only_requested_cells_are_requeued(self):
        self.assertEqual(selected_cells('pipege:lj_dot,pipege:tw_dot,ge2:wk_distmult'),
                         [('pipege', 'lj_dot'), ('pipege', 'tw_dot'), ('ge2', 'wk_distmult')])
        self.assertEqual(len(selected_cells(None)), 12)
        for value in ('', 'ge2:nope', 'ge2:lj_dot,ge2:lj_dot', 'lj_dot'):
            with self.assertRaises(ValueError):
                selected_cells(value)

    def test_bounded_tails_and_no_checkpoints_or_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root/'source'
            src.mkdir()
            (src/'status.json').write_text('{"status":"failed"}')
            (src/'driver.log').write_bytes(b'x'*2048+b'ACTUAL FAILURE')
            (src/'weights.bin').write_bytes(b'weights')
            (src/'external.log').symlink_to(root/'secret.log')
            (root/'secret.log').write_text('not part of run')
            (src/'linked').symlink_to(root, target_is_directory=True)
            report = save_failure_evidence(src, root/'evidence', per_file=256, total=512)
            self.assertEqual({r['path'] for r in report['files']}, {'status.json', 'driver.log'})
            self.assertLessEqual(report['saved_bytes'], 512)
            self.assertTrue((root/'evidence/driver.log').read_bytes().endswith(b'ACTUAL FAILURE'))
            self.assertTrue(next(r for r in report['files'] if r['path']=='driver.log')['tail_only'])
            self.assertFalse((root/'evidence/weights.bin').exists())
            self.assertEqual((src/'weights.bin').read_bytes(), b'weights')

    def test_total_limit_and_nested_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp)/'source'
            src.mkdir()
            for i in range(8):
                (src/(str(i)+'.log')).write_bytes(b'x'*100)
            report = save_failure_evidence(src, Path(temp)/'evidence', per_file=100, total=250)
            self.assertEqual(report['saved_bytes'], 250)
            self.assertEqual(len(report['skipped']), 5)
            with self.assertRaises(ValueError):
                save_failure_evidence(src, src/'nested')


class Monitoring(unittest.TestCase):
    def test_probe_retry_records_exact_command(self):
        events = []
        command = ['squeue', '-h']
        with patch('arc_job_support.subprocess.check_output', side_effect=[
                subprocess.TimeoutExpired(command, 20), 'RUNNING c30\n']) as call:
            self.assertEqual(probe(command, time.monotonic()+60, {}, events), 'RUNNING c30\n')
            self.assertEqual(call.call_count, 2)
        self.assertEqual(events[0]['command'], command)

    def test_exhausted_probe_is_not_workload_timeout(self):
        command = ['nvidia-smi']
        with patch('arc_job_support.subprocess.check_output', side_effect=
                subprocess.TimeoutExpired(command, 20)) as call:
            with self.assertRaisesRegex(RuntimeError, 'Monitoring query failed'):
                probe(command, time.monotonic()+100, {}, [])
            self.assertEqual(call.call_count, 3)

    def test_probe_cannot_extend_deadline(self):
        with patch('arc_job_support.subprocess.check_output') as call:
            with self.assertRaises(AllocationDeadline):
                probe(['squeue'], time.monotonic()-1, {}, [])
            call.assert_not_called()

    def test_recovered_gap_is_not_clean_timing(self):
        with self.assertRaisesRegex(ValueError, 'Monitoring gaps'):
            hardware_check([dict(probe_errors=[dict(command=['squeue'])])], 300, 'GPU-test')

    def test_monitor_failure_cleans_up_real_child_and_preserves_cause(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'train.log'
            env = dict(os.environ, SLURM_JOB_ID='test')
            with patch('arc_job_support.subprocess.check_output', side_effect=
                    subprocess.TimeoutExpired(['squeue'], 20)):
                with self.assertRaisesRegex(RuntimeError, 'Monitoring query failed'):
                    run_logged([sys.executable, '-c', 'import time; time.sleep(60)'], env, log, 120)
            report = json.loads(log.with_suffix('.run_status.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(len(report['probe_errors']), 3)
            self.assertIn('squeue', log.read_text())
            with self.assertRaises(ProcessLookupError):
                os.getpgid(report['pid'])

    def test_success_and_actual_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ)
            env.pop('SLURM_JOB_ID', None)
            log = Path(temp)/'success.log'
            self.assertEqual(run_logged([sys.executable, '-c', 'print("done")'], env, log, 10), 0)
            self.assertIn('done', log.read_text())
            timeout = Path(temp)/'deadline.log'
            self.assertEqual(run_logged([sys.executable, '-c', 'import time; time.sleep(60)'],
                                        env, timeout, .1), 124)
            status = json.loads(timeout.with_suffix('.run_status.json').read_text())
            self.assertEqual(status['status'], 'allocation_deadline')
            with self.assertRaises(ProcessLookupError):
                os.getpgid(status['pid'])

    def test_inactive_allocation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'inactive.log'
            with patch('arc_job_support.subprocess.check_output', return_value='PENDING c30'):
                with self.assertRaisesRegex(RuntimeError, 'not RUNNING'):
                    run_logged([sys.executable, '-c', 'import time; time.sleep(60)'],
                               dict(os.environ, SLURM_JOB_ID='test'), log, 120)


if __name__ == '__main__':
    unittest.main()
