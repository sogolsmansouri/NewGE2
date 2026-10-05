import unittest
import tempfile
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch
import numpy as np

from check_fb_multigpu_training_parity import comparison_passed, relay_validation_counts, tensor_comparison, validate_execution_scope
from run_workstation_fb_accuracy_control import audited_input_trace, control_config, control_flags, schedule_flags, evaluation_panel, engine_flags, data_path_flags
from run_workstation_fb_single_gpu_queue import QUEUES


class TrainingParityTests(unittest.TestCase):
    def test_exact_equal_finite_values_pass(self):
        a = torch.tensor([0.0, 1.0, -0.5])
        self.assertTrue(comparison_passed({'weights': tensor_comparison(a, a.clone())}))

    def test_small_difference_is_not_bitwise_parity(self):
        a = torch.tensor([1.0])
        b = a + 1.e-6
        self.assertFalse(comparison_passed({'weights': tensor_comparison(a, b)}))

    def test_identical_nonfinite_values_fail(self):
        a = torch.tensor([float('inf')])
        self.assertFalse(comparison_passed(tensor_comparison(a, a.clone())))

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            tensor_comparison(torch.zeros(2), torch.zeros(3))

    def test_empty_or_incomplete_comparisons_fail(self):
        self.assertFalse(comparison_passed({}))
        self.assertFalse(comparison_passed({'weights': {}}))


class ExecutionScopeTests(unittest.TestCase):
    def test_exact_authorized_workstation_needs_no_slurm(self):
        with mock.patch('check_fb_multigpu_training_parity.os.uname', return_value=SimpleNamespace(nodename='host.example')), \
             mock.patch('check_fb_multigpu_training_parity.subprocess.check_output') as check:
            validate_execution_scope(None, 'host.example')
            check.assert_not_called()

    def test_other_workstation_is_rejected(self):
        with mock.patch('check_fb_multigpu_training_parity.os.uname', return_value=SimpleNamespace(nodename='other.example')):
            with self.assertRaises(RuntimeError):
                validate_execution_scope(None, 'host.example')

    def test_slurm_requires_running_owned_job_on_this_node(self):
        with mock.patch('check_fb_multigpu_training_parity.os.uname', return_value=SimpleNamespace(nodename='c25.example')), \
             mock.patch.dict('os.environ', USER='owner'), \
             mock.patch('check_fb_multigpu_training_parity.subprocess.check_output',
                        return_value='JobState=RUNNING UserId=owner(1) NodeList=c25 '):
            validate_execution_scope('301209', None)

    def test_wrong_owner_node_or_state_is_rejected(self):
        for allocation in ['JobState=PENDING UserId=owner(1) NodeList=c25 ',
                           'JobState=RUNNING UserId=other(2) NodeList=c25 ',
                           'JobState=RUNNING UserId=owner(1) NodeList=c26 ']:
            with self.subTest(allocation=allocation), \
                 mock.patch('check_fb_multigpu_training_parity.os.uname', return_value=SimpleNamespace(nodename='c25.example')), \
                 mock.patch.dict('os.environ', USER='owner'), \
                 mock.patch('check_fb_multigpu_training_parity.subprocess.check_output', return_value=allocation):
                with self.assertRaises(RuntimeError):
                    validate_execution_scope('301209', None)


class RelayValidationTests(unittest.TestCase):
    def test_negative_hashes_are_valid_coverage(self):
        text = '[stateflow-peer-validate 0] pending_key=-123 dst_mismatch_values=0 src_mismatch_values=0\n'
        self.assertEqual(relay_validation_counts(text), dict(peer_checks=1, negative_handoff_key_checks=1,
                                                           validation_mismatch_lines=0))

    def test_mismatch_count_is_per_copy_not_per_tensor(self):
        text = '[stateflow-peer-validate 0] pending_key=123 dst_mismatch_values=8 src_mismatch_values=8\n'
        self.assertEqual(relay_validation_counts(text), dict(peer_checks=1, negative_handoff_key_checks=0,
                                                           validation_mismatch_lines=1))

    def test_missing_validation_cannot_pass_as_coverage(self):
        self.assertEqual(relay_validation_counts(''), dict(peer_checks=0, negative_handoff_key_checks=0,
                                                         validation_mismatch_lines=0))


class FullControlFlagsTests(unittest.TestCase):
    def setUp(self):
        self.flags = dict(GEGE_BASELINE_TRAINING_SEMANTICS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1',
                          GEGE_BOUNDED_COVER_EPOCH_RELABEL='1', GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS='1',
                          GEGE_BOUNDED_COVER_RELABEL_SEED='17', GEGE_TRAINING_REPLAY_SEED='17')

    def test_no_hidden_control_preserves_optimized_gradient_recipe(self):
        flags = control_flags(self.flags, 16)
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS'], '1')
        self.assertEqual(flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'], '1')
        self.assertEqual(flags['GEGE_BOUNDED_COVER_RELABEL_SEED'], '17')
        self.assertEqual(flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '0')
        self.assertEqual(flags['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'], '0')
        self.assertNotIn('GEGE_TRAINING_REPLAY_SEED', flags)
        self.assertIn('GEGE_TRAINING_REPLAY_SEED', self.flags)

    def test_p32_preserves_pipeline(self):
        flags = control_flags(self.flags, 32)
        self.assertEqual(flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '6')
        self.assertEqual(flags['GEGE_FRAME_CACHE_MAX_STALE_BACKLOG'], '3')
        self.assertEqual(flags['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'], '1')
        self.assertEqual(flags['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'], '1')
        self.assertEqual(flags['GEGE_TRAINING_PARAMETER_AUDIT'], '0')

    def test_p32_sync_changes_only_movement_flags(self):
        flags = control_flags(self.flags, 32, False)
        self.assertEqual(flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '0')
        self.assertEqual(flags['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'], '0')
        self.assertEqual(flags['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'], '0')
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS'], '1')
        self.assertEqual(flags['GEGE_BOUNDED_COVER_EPOCH_RELABEL'], '1')

    def test_explicit_pipeline_is_not_derived_from_partition_count(self):
        flags = control_flags(self.flags, 16, True)
        self.assertEqual(flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'], '6')
        self.assertEqual(flags['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'], '1')

    def test_autograd_reference_preserves_sampling_and_loss(self):
        flags = control_flags(self.flags, 32, False, 'autograd')
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS'], '0')
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS'], '0')
        self.assertEqual(flags['GEGE_BASELINE_TRAINING_SEMANTICS'], '1')
        self.assertEqual(flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'], '1')
        self.assertEqual(flags['GEGE_BOUNDED_COVER_RELABEL_SEED'], '17')
        self.assertEqual(self.flags['GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS'], '1')

    def test_unknown_gradient_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            control_flags(self.flags, 32, False, 'unknown')

    def test_manual_reference_enables_both_model_kernels(self):
        flags = control_flags(self.flags, 32, True, 'manual')
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS'], '1')
        self.assertEqual(flags['GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS'], '1')

    def test_old_weighted_recipe_is_rejected(self):
        self.flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'] = '8'
        with self.assertRaises(ValueError):
            control_flags(self.flags, 32)

    def test_unplanned_partition_count_is_rejected(self):
        with self.assertRaises(ValueError):
            control_flags(self.flags, 20)


class PartitionScheduleControlTests(unittest.TestCase):
    def test_bounded_control_preserves_flags_without_mutation(self):
        flags = dict(GEGE_BOUNDED_GREEDY_COVER_Q4='1', GEGE_STATEFLOW_MAX_ADMITS='3')
        result = schedule_flags(flags, 'bounded')
        self.assertEqual(result, flags)
        self.assertIsNot(result, flags)

    def test_legacy_schedule_clears_all_alternate_ordering_routes(self):
        flags = dict(GEGE_BOUNDED_GREEDY_COVER='1', GEGE_BOUNDED_GREEDY_COVER_Q4='1',
                     GEGE_BOUNDED_COVER_EPOCH_RELABEL='1', GEGE_STATEFLOW_PLANNER='1',
                     GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM='1', GEGE_BOUNDED_STATE_ORDER_FILE='/cover',
                     GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1',
                     GEGE_STATEFLOW_MAX_ADMITS='3', GEGE_GLOBAL_DEGREE_SAMPLING='0')
        result = schedule_flags(flags, 'legacy-random')
        for key in ('GEGE_BOUNDED_GREEDY_COVER', 'GEGE_BOUNDED_GREEDY_COVER_Q4',
                    'GEGE_BOUNDED_COVER_EPOCH_RELABEL', 'GEGE_BOUNDED_Q4_OPTIMAL88',
                    'GEGE_BOUNDED_GREEDY_COVER_REVERSE', 'GEGE_OPTIMIZED_CUSTOM_SCHEDULE',
                    'GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING', 'GEGE_HYBRID_COVER',
                    'GEGE_STATEFLOW_PLANNER', 'GEGE_STATEFLOW_LANE_MATCHING',
                    'GEGE_ACCESS_AWARE_STATE_GENERATION', 'GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM'):
            self.assertEqual(result[key], '0', key)
        self.assertNotIn('GEGE_BOUNDED_STATE_ORDER_FILE', result)
        self.assertEqual(result['GEGE_STATEFLOW_MAX_ADMITS'], '4')
        for key in ('GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS', 'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE',
                    'GEGE_GLOBAL_DEGREE_SAMPLING'):
            self.assertEqual(result[key], flags[key])
        self.assertEqual(flags['GEGE_BOUNDED_GREEDY_COVER'], '1')
        self.assertEqual(flags['GEGE_BOUNDED_STATE_ORDER_FILE'], '/cover')

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            schedule_flags({}, 'unknown')


class DataPathControlTests(unittest.TestCase):
    def test_default_preserves_existing_controls(self):
        original = dict(GEGE_FAST_MAP_TENSORS='1', GEGE_FIXED_BUFFER_MASKED_UPDATE='1')
        result = data_path_flags(original, 'current')
        self.assertEqual(result, original)
        self.assertIsNot(result, original)

    def test_reference_disables_fast_paths_without_changing_objective(self):
        original = dict(GEGE_FAST_MAP_TENSORS='1', GEGE_FIXED_BUFFER_MASKED_UPDATE='1',
                        GEGE_BASELINE_TRAINING_SEMANTICS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1',
                        GEGE_GLOBAL_DEGREE_SAMPLING='0', GEGE_FRAME_CACHE_HIDDEN_FRAMES='0')
        result = data_path_flags(original, 'reference')
        self.assertEqual(result['GEGE_UNIQUE_BACKEND'], 'sort')
        self.assertEqual(result['GEGE_SYNC_BEFORE_SWAP'], '1')
        for key in ('GEGE_FAST_MAP_TENSORS', 'GEGE_FIXED_BUFFER_BITMAP_MAP',
                    'GEGE_FIXED_BUFFER_MASKED_UPDATE', 'GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS',
                    'GEGE_PARTITION_BUFFER_LP_FAST_PATH', 'GEGE_GPU_ACTIVE_EDGE_SHUFFLE'):
            self.assertEqual(result[key], '0')
        for key in ('GEGE_BASELINE_TRAINING_SEMANTICS', 'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE',
                    'GEGE_GLOBAL_DEGREE_SAMPLING', 'GEGE_FRAME_CACHE_HIDDEN_FRAMES'):
            self.assertEqual(result[key], original[key])
        self.assertEqual(original['GEGE_FAST_MAP_TENSORS'], '1')

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            data_path_flags({}, 'unknown')


class SingleGpuConfigTests(unittest.TestCase):
    def setUp(self):
        self.template = dict(model=dict(decoder=dict(type='DISTMULT'), encoder=dict(embedding_dim=100)),
                             storage=dict(dataset={}, embeddings=dict(options={})),
                             training=dict(batch_size=50000, negative_sampling=dict(
                                 degree_fraction=0.5, negatives_per_positive=1000, num_chunks=50)), evaluation={})

    def test_complex_has_matching_training_decoder_and_frozen_workload(self):
        config = control_config(self.template, 'complex', 32, 4, Path('/data'), Path('/model'))
        self.assertEqual(config['model']['decoder']['type'], 'COMPLEX')
        self.assertEqual(config['training']['batch_size'], 50000)
        self.assertEqual(config['training']['num_epochs'], 10)
        self.assertEqual(config['storage']['device_ids'], [0])
        self.assertEqual(config['storage']['embeddings']['options'], dict(num_partitions=32, buffer_capacity=4))
        self.assertEqual(self.template['model']['decoder']['type'], 'DISTMULT')
        self.assertNotIn('num_epochs', self.template['training'])

    def test_unplanned_decoder_or_workload_is_rejected(self):
        with self.assertRaises(ValueError):
            control_config(self.template, 'dot', 32, 4, Path('/data'), Path('/model'))
        self.template['training']['batch_size'] = 150000
        with self.assertRaises(ValueError):
            control_config(self.template, 'complex', 32, 4, Path('/data'), Path('/model'))

    def test_early_control_changes_only_epoch_count(self):
        full = control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'))
        early = control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'), 3)
        self.assertEqual(early['training']['num_epochs'], 3)
        self.assertEqual(early['training']['batch_size'], 50000)
        early['training']['num_epochs'] = 10
        self.assertEqual(early, full)
        with self.assertRaises(ValueError):
            control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'), 0)

    def test_sampling_intervention_changes_only_requested_fraction(self):
        original = control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'), 3)
        for fraction in (0.0, 0.5, 1.0):
            changed = control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'), 3, fraction)
            self.assertEqual(changed['training']['negative_sampling']['degree_fraction'], fraction)
            changed['training']['negative_sampling']['degree_fraction'] = 0.5
            self.assertEqual(changed, original)
        self.assertEqual(self.template['training']['negative_sampling']['degree_fraction'], 0.5)

    def test_invalid_sampling_intervention_is_rejected(self):
        for fraction in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                control_config(self.template, 'distmult', 32, 4, Path('/data'), Path('/model'), 3, fraction)

    def test_released_engine_has_no_optimized_feature_flags(self):
        flags = dict(GEGE_FIXED_BUFFER_BITMAP_MAP='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1')
        self.assertEqual(engine_flags(flags, 'zenodo'), {})
        self.assertEqual(engine_flags(flags, 'optimized'), flags)
        self.assertIsNot(engine_flags(flags, 'optimized'), flags)
        self.assertEqual(flags['GEGE_FIXED_BUFFER_BITMAP_MAP'], '1')
        with self.assertRaises(ValueError):
            engine_flags(flags, 'unknown')

    def test_replay_trace_is_order_independent_but_not_content_independent(self):
        a = '[training-input] epoch=0 batch=0 edges=a\n[training-input] epoch=0 batch=1 edges=b\n'
        b = '\n'.join(reversed(a.splitlines()))
        self.assertEqual(audited_input_trace(a), audited_input_trace(b))
        self.assertNotEqual(audited_input_trace(a), audited_input_trace(a.replace('edges=a', 'edges=c')))
        self.assertEqual(audited_input_trace(a)['batches'], 2)

    def test_absent_trace_is_not_successful_audit(self):
        with self.assertRaises(ValueError):
            audited_input_trace('Training completed')

    def test_queued_controls_keep_partition_and_execution_comparisons_separate(self):
        cases = [case for queue in QUEUES.values() for case in queue]
        self.assertEqual(len({case[0] for case in cases}), 4)
        self.assertEqual({case[1] for case in cases}, {16, 32})
        self.assertIn(('p32_pipeline_manual', 32, 'on', 'manual'), cases)
        self.assertIn(('p32_sync_manual', 32, 'off', 'manual'), cases)
        self.assertIn(('p32_sync_autograd', 32, 'off', 'autograd'), cases)


class DiagnosticQueryPanelTests(unittest.TestCase):
    def test_fixed_seed_selects_same_nonprefix_panel_for_both_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'queries.bin'
            rows = np.arange(30000, dtype='<i4').reshape(10000, 3)
            rows.tofile(source)
            for name in ('a', 'b', 'c'):
                (root/name).mkdir()
            a, info_a = evaluation_panel(source, 1000, 17, root/'a')
            b, info_b = evaluation_panel(source, 1000, 17, root/'b')
            c, info_c = evaluation_panel(source, 1000, 18, root/'c')
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(info_a['query_sha256'], info_b['query_sha256'])
            self.assertNotEqual(info_a['query_sha256'], info_c['query_sha256'])
            indices = np.fromfile(info_a['selected_indices_path'], dtype='<u8')
            self.assertEqual(len(np.unique(indices)), 1000)
            self.assertTrue(np.all(indices[:-1] < indices[1:]))
            self.assertFalse(np.array_equal(indices, np.arange(1000)))
            self.assertTrue(np.array_equal(np.fromfile(a, dtype='<i4').reshape(-1, 3), rows[indices]))

    def test_full_panel_and_invalid_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'queries.bin'
            np.zeros((10000, 3), dtype='<i4').tofile(source)
            path, info = evaluation_panel(source, 10000, 17, root)
            self.assertEqual(path, source)
            self.assertEqual(info['selection'], 'full_frozen_10000')
            for count, seed in [(0, 17), (10001, 17), (1000, -1)]:
                with self.assertRaises(ValueError):
                    evaluation_panel(source, count, seed, root)
            np.zeros((1000, 3), dtype='<i4').tofile(source)
            with self.assertRaises(ValueError):
                evaluation_panel(source, 1000, 17, root)


if __name__ == '__main__':
    unittest.main()
