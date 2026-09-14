import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tw_visible_frame_plan import plan, NODES, ROW_BYTES


class VisibleFramePlanTest(unittest.TestCase):
    def test_historical_table(self):
        self.assertEqual([plan(4,k,20)["p"] for k in range(4,11)], [7,8,10,11,13,14,16])

    def test_q_changes_hidden_not_payload_at_fixed_k(self):
        rows = [plan(q,7,32) for q in (2,3,4)]
        self.assertEqual([r["hidden"] for r in rows], [5,4,3])
        self.assertEqual({r["p"] for r in rows}, {7})

    def test_bounds_and_padding(self):
        for c in (16,20,24,32,48,80):
            for q in (2,3,4):
                for k in range(q,11):
                    r = plan(q,k,c)
                    self.assertGreaterEqual(r["p"],q)
                    self.assertLessEqual(r["frame_total_gib"],c)
                    if r["p"] > q:
                        import math
                        self.assertGreater(k*math.ceil(NODES/(r["p"]-1))*ROW_BYTES,c*2**30)


if __name__ == "__main__":
    unittest.main()
