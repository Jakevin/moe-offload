"""Model-free unit tests for prompt_cache helpers (run: python3.11 test_prompt_cache.py)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
import prompt_cache as PC


class T(unittest.TestCase):
    def test_plan_chunks(self):
        self.assertEqual(PC.plan_chunks(100, 0, [40], 0), [(0, 40), (40, 100)])
        self.assertEqual(PC.plan_chunks(100, 50, [40], 0), [(50, 100)])
        self.assertEqual(PC.plan_chunks(100, 0, [], 0), [(0, 100)])
        self.assertEqual(PC.plan_chunks(100, 10, [40], 32), [(10, 40), (40, 72), (72, 100)])
        self.assertEqual(PC.plan_chunks(100, 0, [], 40), [(0, 40), (40, 80), (80, 100)])

    def test_anchor(self):
        S, E, NL, SYS = 1, 2, 3, [9]
        self.assertEqual(PC.find_anchor_a1([S, 9, 5, 5, E, NL, S, 7, 8, E, NL], S, E, NL, SYS), 6)
        self.assertIsNone(PC.find_anchor_a1([S, 7, 5, E, NL, S], S, E, NL, SYS))

    def test_copy_exact(self):
        a = mx.array([-0.0, 1.5, float("nan"), float("inf")], dtype=mx.bfloat16)
        b = PC.copy_array(a)
        self.assertEqual(mx.view(a, mx.uint16).tolist(), mx.view(b, mx.uint16).tolist())
        for dt in (mx.float32, mx.float16, mx.int64, mx.int32, mx.uint8, mx.bool_):
            x = mx.array([1, 0, 1], dtype=dt); self.assertEqual(PC.copy_array(x).tolist(), x.tolist())
            self.assertEqual(PC.copy_array(x).dtype, dt)

    def test_snapshot_isolation(self):
        class Obj:  # stand-in for a KV cache object updated in place
            def __init__(self):
                self.keys = mx.zeros((1, 4)); self.offset = 0; self.cache = [mx.ones((2,)), None]
        class LM: _position_ids = None; _rope_deltas = None
        o = Obj(); o.keys[:, 0:2] = mx.array([[1.0, 2.0]]); o.offset = 2
        st, nb = PC.snapshot_state([o], LM)
        o.keys[:, 2:4] = mx.array([[3.0, 4.0]]); o.offset = 4; o.cache[0] = o.cache[0] * 5
        r = PC.restore_state(st, LM)[0]
        self.assertEqual(r.keys.tolist(), [[1.0, 2.0, 0.0, 0.0]]); self.assertEqual(r.offset, 2)
        self.assertEqual(r.cache[0].tolist(), [1.0, 1.0])
        r.keys[:, 2:3] = mx.array([[9.0]])
        r2 = PC.restore_state(st, LM)[0]
        self.assertEqual(r2.keys.tolist(), [[1.0, 2.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
