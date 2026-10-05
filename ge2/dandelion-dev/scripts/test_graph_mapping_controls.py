import unittest

from check_full_reload_graph_mapping import validation_flags


class MappingControlTests(unittest.TestCase):
    def test_production_schedule_and_pipeline_are_preserved(self):
        original = dict(GEGE_BOUNDED_GREEDY_COVER_Q4='1', GEGE_BOUNDED_Q4_OPTIMAL88='1',
                        GEGE_BOUNDED_COVER_EPOCH_RELABEL='1', GEGE_STATEFLOW_MAX_ADMITS='3')
        result = validation_flags(original, 'bounded', True, True)
        for key, value in original.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '6')
        self.assertEqual(result['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'], '1')
        self.assertEqual(result['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'], '1')
        self.assertNotIn('GEGE_FRAME_CACHE_HIDDEN_FRAMES', original)

    def test_full_reload_regression_still_disables_pipeline(self):
        result = validation_flags({'GEGE_FRAME_CACHE_HIDDEN_FRAMES': '6'}, 'legacy-random', False, False)
        self.assertEqual(result['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '0')
        self.assertEqual(result['GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM'], '0')
        self.assertEqual(result['GEGE_BOUNDED_GREEDY_COVER_Q4'], '0')

    def test_unsupported_pipeline_controls_fail_before_running(self):
        for schedule, retain in (('legacy-random', True), ('bounded', False)):
            with self.assertRaises(ValueError):
                validation_flags({}, schedule, retain, True)


if __name__ == '__main__':
    unittest.main()
