import copy
from pathlib import Path
import unittest

from run_arc_tw_fast_replay import EDGES, FAST, check_schedule, check_training, fast_environment, make_config


class FastReplayTest(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(model=dict(random_seed=741135446461071584,
            encoder={'layers': [[{'output_dim': 100}]]}, decoder={'type': 'DISTMULT'},
            loss={'type': 'SOFTMAX_CE', 'options': {'reduction': 'SUM'}},
            dense_optimizer={'type': 'ADAGRAD', 'options': {'learning_rate': .1}},
            sparse_optimizer={'type': 'ADAGRAD', 'options': {'learning_rate': .1}}),
            storage=dict(prefetch=True, embeddings={'options': {'num_partitions':16, 'buffer_capacity':4}}),
            training=dict(batch_size=50000, negative_sampling_method='RNS', negative_sampling=dict(
                num_chunks=50, negatives_per_positive=1000, degree_fraction=.5, superbatch_negative_plan_batches=8)),
            evaluation={})
        self.data = dict(num_nodes=41652230, num_train=EDGES, num_relations=1)

    def test_preserves_fast_training_settings(self):
        original = copy.deepcopy(self.cfg)
        cfg = make_config(self.cfg, self.data, Path('/model'), 10)
        self.assertEqual(self.cfg, original)
        self.assertEqual(cfg['training']['negative_sampling'], original['training']['negative_sampling'])
        self.assertEqual(cfg['storage']['dataset'], self.data)
        self.assertTrue(cfg['training']['save_model'])
        self.assertFalse(make_config(self.cfg, self.data, Path('/gate'), 2)['training']['save_model'])

    def test_rejects_other_workloads(self):
        for field, value in [('batch_size',150000), ('negative_sampling_method','GAN')]:
            cfg = copy.deepcopy(self.cfg)
            cfg['training'][field] = value
            with self.assertRaises(ValueError): make_config(cfg, self.data, Path('/x'), 10)
        self.data['num_train'] = 1468345182
        with self.assertRaises(ValueError): make_config(self.cfg, self.data, Path('/x'), 10)

    def test_flags_restore_fast_path_without_replay_seed(self):
        flags = dict(FAST, PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',
                     GEGE_TRAINING_REPLAY_SEED='7', GEGE_TRAINING_INPUT_AUDIT='1')
        output = fast_environment(flags, Path('/order'), False)
        for key, value in FAST.items(): self.assertEqual(output[key], value)
        self.assertNotIn('GEGE_TRAINING_REPLAY_SEED', output)
        self.assertNotIn('GEGE_TRAINING_INPUT_AUDIT', output)
        self.assertEqual(output['GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY'], '0')
        self.assertEqual(fast_environment(flags, Path('/order'), True)['GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY'], '1')
        flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'] = '6'
        with self.assertRaises(ValueError): fast_environment(flags, Path('/order'), False)

    def log(self):
        return ('deferred backing allocation device=cuda:0 visible_rows=10413060 physical_rows=18222855 dim=100 pinned=true hidden_frames=3\n' * 2
                + 'Generating bounded GREEDY_COVER ordering states=20\n[manual_dot_rns] enabled=1\nUsing bucket-streaming LP path\n'
                + ''.join(f'Edges processed: [{EDGES}/{EDGES}], 100.00%\nFinished training epoch {i}\nEpoch Runtime: 200000ms\n'
                          '[frame_cache] swap_samples=19 visible_install_parts=0 hidden_publish_parts=57 hidden_publish_rows=1 fallback_visible_admit_parts=0 preload_miss_swaps=0\n'
                          for i in (1,2)))

    def test_complete_training(self):
        self.assertEqual(check_training(self.log(), 2), [200,200])

    def test_rejects_incomplete_or_wrong_path(self):
        for old, new in [('physical_rows=18222855','physical_rows=26032650'), ('states=20','states=35'),
                         ('hidden_publish_parts=57','hidden_publish_parts=54'), ('preload_miss_swaps=0','preload_miss_swaps=1'),
                         ('Finished training epoch 2','Finished training epoch 1'), ('Epoch Runtime: 200000ms','Epoch Runtime: 0ms'),
                         ('[manual_dot_rns] enabled=1','manual off')]:
            with self.subTest(old=old), self.assertRaises(ValueError): check_training(self.log().replace(old,new),2)

    def test_incomplete_schedule_rejected(self):
        with self.assertRaises(ValueError): check_schedule('state=[0,1,2,3]\n')
        with self.assertRaises(ValueError): check_schedule('state=[0,0,2,3]\n')


if __name__ == '__main__':
    unittest.main()
