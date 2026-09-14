import sys
import unittest
import tempfile
import signal
import yaml
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_arc_tw_memory_study import (study_rows, all_resident_payload, relocate_dataset_metadata,
    reusable_results, interrupt_allocation, AllocationInterrupted)
from run_local_tw_fixed_frames import NODES, TRAIN, digest, summarize, write_json


class ArcMemoryStudyTest(unittest.TestCase):
    def test_sigterm_is_a_catchable_allocation_interruption(self):
        with self.assertRaises(AllocationInterrupted) as caught:
            interrupt_allocation(signal.SIGTERM, None)
        self.assertEqual(caught.exception.signum, signal.SIGTERM)
        self.assertEqual(caught.exception.code, 143)

    def test_resume_revalidates_full_cases_and_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory)
            full = previous / 'case/full'
            artifacts = full.parent / 'artifacts'
            full.mkdir(parents=True)
            artifacts.mkdir()
            (artifacts / 'gege_train').write_bytes(b'frozen binary')
            hashes = {'gege_train': digest(artifacts / 'gege_train')}
            row = dict(case='case', p=7, q=4, k=4, hp=0, hs=0, states=7, overlap=6)
            log = ('[fixed-frame-budget] visible=4 preload=0 stale=0 physical=4 '
                   'partition_rows=1 width=100 bytes=3200\n') * 2
            log += 'Ordering states=7 transitions=6 total_buckets=49 max_admits=3 transition_admits=18\n'
            log += (f'Edges processed: [{TRAIN}/{TRAIN}]\nEpoch Runtime: 1000ms\n') * 5
            (full / 'train.log').write_text(log)
            (full / 'effective_config.yaml').write_text('training: {batch_size: 50000}\n')
            write_json(full / 'provenance.json', {'config_sha256': digest(full / 'effective_config.yaml')})
            write_json(full / 'result.json', summarize(log, row, 5, 0, []))
            entry = dict(row, status='valid_timing', run_dir=str(full))
            write_json(previous / 'results.json', [entry, dict(case='unfinished', status='failed_or_invalid')])
            write_json(previous / 'plan.json', dict(commit='commit', batch_size=50000, epochs=5))
            write_json(previous / 'build.json', dict(commit='commit', hashes=hashes))
            accepted = reusable_results(previous, [row], 'commit', hashes)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]['steady_epoch_s'], 1)
            self.assertEqual(accepted[0]['reused_from'], str(previous))
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                reusable_results(previous, [row], 'different', hashes)
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                reusable_results(previous, [dict(row, k=5)], 'commit', hashes)
            (full / 'train.log').write_text(log.replace(f'Edges processed: [{TRAIN}/{TRAIN}]', '', 1))
            with self.assertRaisesRegex(RuntimeError, 'log revalidation'):
                reusable_results(previous, [row], 'commit', hashes)
            (full / 'train.log').write_text(log)
            (artifacts / 'gege_train').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'binary changed'):
                reusable_results(previous, [row], 'commit', hashes)

    def test_relocation_preserves_counts_and_original_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)
            original='dataset_dir: /old/machine/\nnum_train: 42\nnum_nodes: 16\n'
            (data/'dataset.yaml').write_text(original)
            relocate_dataset_metadata(data)
            new=yaml.safe_load((data/'dataset.yaml').read_text())
            self.assertEqual(new,dict(dataset_dir=str(data.resolve())+'/',num_train=42,num_nodes=16))
            backup=list(data.glob('dataset.before_relocation.*.yaml'))
            self.assertEqual(len(backup),1)
            self.assertEqual(backup[0].read_text(),original)
            relocate_dataset_metadata(data)
            self.assertEqual(len(list(data.glob('dataset.before_relocation.*.yaml'))),1)

    def test_every_k_and_original_quota_split_remain_in_plan(self):
        rows = study_rows()
        self.assertEqual(len(rows),21)
        self.assertEqual(len({r['case'] for r in rows}),21)
        self.assertEqual({r['q'] for r in rows},{4})
        self.assertEqual({r['k'] for r in rows},set(range(4,11)))
        q4shared=[r for r in rows if r.get('shared_hidden') and r['q']==4]
        self.assertEqual({r['k'] for r in q4shared},set(range(5,11)))
        self.assertEqual({r['p'] for r in q4shared if r['k']==7},{8,10,11,16})
        fixed=[r for r in rows if not r.get('shared_hidden') and r['q']==4 and r['k']==5]
        self.assertEqual({(r['hp'],r['hs']) for r in fixed},{(0,1),(1,0)})

    def test_all_resident_counts_graph_not_just_entity(self):
        r=all_resident_payload(NODES,TRAIN,50897289216)
        self.assertEqual(r['status'],'infeasible_payload')
        self.assertLess(r['entity_bytes'],r['allowed_bytes'])
        self.assertGreater(r['minimum_payload_bytes'],r['capacity_bytes'])
        self.assertEqual(all_resident_payload(NODES,TRAIN,80*2**30)['status'],'requires_runtime_memory_gate')
        self.assertEqual(r['effective_visible_frames'],2)
        self.assertEqual(r['hidden_frames'],0)


if __name__=='__main__':
    unittest.main()
