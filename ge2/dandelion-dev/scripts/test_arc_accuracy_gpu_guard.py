import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from arc_accuracy_gpu_guard import foreign_gpu_pids, guarded_run


class GpuGuardTests(unittest.TestCase):
    def test_only_own_process_group_is_allowed(self):
        with patch('arc_accuracy_gpu_guard.subprocess.check_output', return_value='11\n12\n13\n'), \
             patch('arc_accuracy_gpu_guard.os.getpgid', side_effect=[100, 200, ProcessLookupError()]):
            self.assertEqual(foreign_gpu_pids(0, 100), [12])

    def test_busy_gpu_rejects_before_launch(self):
        with patch('arc_accuracy_gpu_guard.foreign_gpu_pids', return_value=[123]), \
             patch('arc_accuracy_gpu_guard.subprocess.Popen') as child:
            with self.assertRaisesRegex(RuntimeError, 'before launch'):
                guarded_run([], {}, Path('/unused'), 10, 0, Path('/unused'))
            child.assert_not_called()

    def test_contention_kills_only_the_own_child(self):
        foreign = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)
        popen, children = subprocess.Popen, []
        def launch(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            return child
        try:
            with tempfile.TemporaryDirectory() as temp, \
                 patch('arc_accuracy_gpu_guard.foreign_gpu_pids', side_effect=[[], [foreign.pid]]), \
                 patch('arc_accuracy_gpu_guard.subprocess.check_output', return_value='RUNNING '+os.uname().nodename.split('.')[0]), \
                 patch('arc_accuracy_gpu_guard.subprocess.Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'during accuracy control'):
                    guarded_run(['/bin/sleep', '30'], dict(os.environ, SLURM_JOB_ID='test'),
                                Path(temp)/'log', 10, 0, Path(temp)/'monitor')
                self.assertIsNotNone(children[0].poll())
                self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)

    def test_success_and_deadline(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('arc_accuracy_gpu_guard.foreign_gpu_pids', return_value=[]), \
             patch('arc_accuracy_gpu_guard.subprocess.check_output', return_value='RUNNING '+os.uname().nodename.split('.')[0]):
            args = (['/bin/true'], dict(os.environ, SLURM_JOB_ID='test'), Path(temp)/'log')
            self.assertEqual(guarded_run(*args, 10, 0, Path(temp)/'monitor'), 0)
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                guarded_run(*args, 0, 0, Path(temp)/'monitor')


if __name__ == '__main__':
    unittest.main()
