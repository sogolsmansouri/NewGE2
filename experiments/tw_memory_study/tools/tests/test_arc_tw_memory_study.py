import sys
import unittest
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_arc_tw_memory_study import study_rows, all_resident_payload, relocate_dataset_metadata
from run_local_tw_fixed_frames import NODES, TRAIN


class ArcMemoryStudyTest(unittest.TestCase):
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
