from pathlib import Path
import tempfile
import unittest

from run_arc_multigpu_diagnostic import stage_scripts


class DiagnosticStagingTests(unittest.TestCase):
    def test_stages_launchers_without_following_historical_package_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            (source/'launcher.py').write_text('pass\n')
            (source/'gege_pkg').mkdir()
            (source/'gege_pkg/gege').symlink_to(root/'missing-package')
            stage_scripts(source, root/'payload')
            self.assertEqual((root/'payload/launcher.py').read_text(), 'pass\n')
            self.assertEqual(list((root/'payload').iterdir()), [root/'payload/launcher.py'])
            with self.assertRaises(FileExistsError):
                stage_scripts(source, root/'payload')

    def test_does_not_accept_linked_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            (source/'launcher.py').symlink_to(root/'missing')
            with self.assertRaisesRegex(ValueError, 'regular source files'):
                stage_scripts(source, root/'payload')


if __name__ == '__main__':
    unittest.main()
