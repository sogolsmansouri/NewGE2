import json
from pathlib import Path
import tempfile
import unittest

from run_arc_multigpu_control import CASES, verify_payload
from run_arc_paper_case import digest


class MultiGpuControlTests(unittest.TestCase):
    def test_scope_is_four_gpu_tw_and_fb_for_both_systems(self):
        self.assertEqual(set(CASES), {f'{system}_{workload}_4gpu'
            for system in ('pipege', 'ge2') for workload in ('tw_dot', 'fb_complex')})

    def test_payload_requires_frozen_files_and_power_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = root/'runner.py'
            program.write_text('pass\n')
            metadata = dict(power_w=200, files={'runner.py': digest(program)})
            manifest = root/'payload_manifest.json'

            def freeze():
                manifest.write_text(json.dumps(metadata))
                (root/'READY').write_text(digest(manifest)+'\n')

            freeze()
            self.assertEqual(verify_payload(root), metadata)
            program.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'file changed'):
                verify_payload(root)
            program.write_text('pass\n')
            metadata['power_w'] = 250
            freeze()
            with self.assertRaisesRegex(ValueError, 'power cohort'):
                verify_payload(root)
            (root/'READY').write_text('wrong\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_payload(root)


if __name__ == '__main__':
    unittest.main()
