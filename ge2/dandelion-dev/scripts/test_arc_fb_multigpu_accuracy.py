import unittest

from run_arc_fb_multigpu_accuracy import control_flags


class FbTransportTests(unittest.TestCase):
    def test_changes_only_transport_between_replayed_controls(self):
        reference = dict(GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS='1',
                         GEGE_STATEFLOW_PEER_RUNTIME='on',
                         GEGE_FRAME_CACHE_HIDDEN_FRAMES='6',
                         GEGE_BOUNDED_COVER_EPOCH_RELABEL='1')
        peer = control_flags(reference, 'peer')
        host = control_flags(reference, 'host')
        self.assertEqual({key for key in peer if peer[key] != host[key]},
                         {'GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK'})
        self.assertEqual(host['GEGE_STATEFLOW_PEER_RUNTIME'], 'on')
        self.assertEqual(host['GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS'], '1')
        self.assertEqual(peer['GEGE_TRAINING_REPLAY_SEED'], '17')
        self.assertEqual(peer['GEGE_TRAINING_INPUT_AUDIT'], '1')
        self.assertNotIn('GEGE_TRAINING_REPLAY_SEED', reference)

    def test_unknown_transport_fails(self):
        with self.assertRaises(ValueError):
            control_flags({}, 'off')


if __name__ == '__main__':
    unittest.main()
