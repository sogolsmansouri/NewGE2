import copy
from pathlib import Path
import unittest

from run_zenodo_fb_sampling_control import case_config, check_reference, training_times, conditions


def reference():
    optimizer = dict(type='ADAGRAD', options=dict(learning_rate=.1))
    return dict(model=dict(decoder=dict(type='DISTMULT', options=dict(input_dim=100, inverse_edges=True)),
                           encoder=dict(layers=[[dict(type='EMBEDDING', output_dim=100,
                                                     init=dict(type='GLOROT_UNIFORM'))]]),
                           dense_optimizer=copy.deepcopy(optimizer), sparse_optimizer=copy.deepcopy(optimizer),
                           random_seed=741135446461071584,
                           loss=dict(type='SOFTMAX_CE', options=dict(reduction='SUM'))),
                storage=dict(dataset=dict(dataset_dir='/old/data/'), model_dir='/old/model/',
                             checkpoint_dir='/old/model/', embeddings=dict(type='MEM_PARTITION_BUFFER',
                                 options=dict(num_partitions=16, buffer_capacity=4))),
                training=dict(batch_size=50000, num_epochs=10, negative_sampling_method='RNS',
                              negative_sampling=dict(degree_fraction=.5, num_chunks=50, negatives_per_positive=1000)),
                evaluation=dict(epochs_per_eval=11))


class SamplingControlTests(unittest.TestCase):
    def test_seed_controls_change_only_seed(self):
        self.assertEqual(conditions('seeds'), [('seed42', .5, None), ('seed123', .5, None)])
        for seed in (42, 123):
            original = reference()
            config = case_config(original, Path('/old/data'), Path('/old/model'), .5, seed=seed)
            self.assertEqual(config['model']['random_seed'], seed)
            config['model']['random_seed'] = original['model']['random_seed']
            self.assertEqual(config, original)
        with self.assertRaises(ValueError):
            case_config(reference(), Path('/data'), Path('/model'), .5, seed=999)
        with self.assertRaises(ValueError):
            case_config(reference(), Path('/data'), Path('/model'), .5, 'normal_0001', seed=42)

    def test_initialization_changes_only_one_learning_factor(self):
        original = reference()
        untouched = copy.deepcopy(original)
        config = case_config(original, Path('/old/data'), Path('/old/model'), .5, 'normal_0001')
        self.assertEqual(original, untouched)
        init = config['model']['encoder']['layers'][0][0]['init']
        self.assertEqual(init, dict(type='NORMAL', options=dict(mean=0., std=.001)))
        config['model']['encoder']['layers'][0][0]['init'] = dict(type='GLOROT_UNIFORM')
        self.assertEqual(config, original)
        self.assertEqual(conditions('initialization'), [('normal_0001', .5, None)])
        with self.assertRaises(ValueError):
            case_config(original, Path('/data'), Path('/model'), .5, 'choose_by_test_mrr')

    def test_repartition_preserves_learning_config(self):
        self.assertEqual(conditions('repartition'), [('fixed', .5, 'fixed'), ('repartition', .5, 'repartition')])
        a, b = [case_config(reference(), Path('/data'), Path('/model'), fraction)
                for _, fraction, _ in conditions('repartition')]
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            conditions('best_test_score')

    def test_reference(self):
        check_reference(reference(), 'distmult')

    def test_wrong_reference(self):
        for key, value in [('num_epochs', 5), ('batch_size', 10000), ('negative_sampling_method', 'GAN')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                config = reference()
                config['training'][key] = value
                check_reference(config, 'distmult')
        with self.assertRaises(ValueError):
            check_reference(reference(), 'complex')

    def test_only_paths_and_degree_change(self):
        original = reference()
        untouched = copy.deepcopy(original)
        config = case_config(original, Path('/new/data'), Path('/new/model'), 0.)
        self.assertEqual(original, untouched)
        config['storage']['dataset'] = original['storage']['dataset']
        for name in ('model_dir', 'checkpoint_dir'):
            config['storage'][name] = original['storage'][name]
        config['training']['negative_sampling']['degree_fraction'] = .5
        self.assertEqual(config, original)

    def test_prespecified_factors_only(self):
        with self.assertRaises(ValueError):
            case_config(reference(), Path('/data'), Path('/model'), .25)

    def test_epoch_validation(self):
        log = '\n'.join(f'Edges processed: [304727650/304727650]\n'
                        f'Finished training epoch {i}\nEpoch Runtime: 250000ms' for i in range(1, 11))
        self.assertEqual(training_times(log), [250.] * 10)
        for bad in (log.replace('epoch 10', 'epoch 9'), log.replace('304727650', '304727649'),
                    log.replace('Epoch Runtime: 250000ms', '', 1)):
            with self.subTest(log=bad[-60:]), self.assertRaises(ValueError):
                training_times(bad)


if __name__ == '__main__':
    unittest.main()
