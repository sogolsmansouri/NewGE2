import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch

from check_fb_multigpu_training_parity import comparison_passed, relay_validation_counts, tensor_comparison, validate_execution_scope
from run_workstation_fb_accuracy_control import audited_input_trace, control_config, control_flags


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


class SingleGpuConfigTests(unittest.TestCase):
    def setUp(self):
        self.template = dict(model=dict(decoder=dict(type='DISTMULT'), encoder=dict(embedding_dim=100)),
                             storage=dict(dataset={}, embeddings=dict(options={})),
                             training=dict(batch_size=50000), evaluation={})

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

    def test_replay_trace_is_order_independent_but_not_content_independent(self):
        a = '[training-input] epoch=0 batch=0 edges=a\n[training-input] epoch=0 batch=1 edges=b\n'
        b = '\n'.join(reversed(a.splitlines()))
        self.assertEqual(audited_input_trace(a), audited_input_trace(b))
        self.assertNotEqual(audited_input_trace(a), audited_input_trace(a.replace('edges=a', 'edges=c')))
        self.assertEqual(audited_input_trace(a)['batches'], 2)

    def test_absent_trace_is_not_successful_audit(self):
        with self.assertRaises(ValueError):
            audited_input_trace('Training completed')


if __name__ == '__main__':
    unittest.main()
