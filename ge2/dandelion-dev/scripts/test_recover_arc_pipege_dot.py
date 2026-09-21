import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from recover_arc_pipege_dot import require_equal, verify_checkpoint_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root/'model'
        self.checkpoint.mkdir()
        self.saved = dict(source=str(self.checkpoint), files=[])
        for name in ('embeddings.bin', 'embeddings_state.bin', 'model.pt_0'):
            path = self.checkpoint/name
            path.write_bytes(b'checkpoint fixture')
            self.saved['files'].append(dict(path=name, bytes=path.stat().st_size, sha256=digest(path)))

    def test_unchanged_checkpoint(self):
        verify_checkpoint_files(self.checkpoint, self.saved, digest)

    def test_changed_checkpoint_rejected(self):
        (self.checkpoint/'embeddings.bin').write_bytes(b'changed checkpoint')
        with self.assertRaises(ValueError):
            verify_checkpoint_files(self.checkpoint, self.saved, digest)

    def test_missing_and_duplicate_members_rejected(self):
        for entries in (self.saved['files'][1:], self.saved['files'] + self.saved['files'][:1]):
            saved = dict(self.saved, files=entries)
            with self.assertRaises(ValueError):
                verify_checkpoint_files(self.checkpoint, saved, digest)

    def test_symlink_escape_rejected(self):
        path = self.checkpoint/'embeddings.bin'
        target = self.root/'outside.bin'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            verify_checkpoint_files(self.checkpoint, self.saved, digest)

    def test_relocated_checkpoint_rejected(self):
        saved = copy.deepcopy(self.saved)
        saved['source'] = str(self.root/'other')
        with self.assertRaises(ValueError):
            verify_checkpoint_files(self.checkpoint, saved, digest)

    def test_contract_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            require_equal('wrong', 'expected', 'Commit')


if __name__ == '__main__':
    unittest.main()
