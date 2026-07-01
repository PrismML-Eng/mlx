"""Validate mx.fast.spec_decode_verify against a pure-Python oracle mirroring the
easymlx-swift verify loop. Dependency-free (mlx + stdlib).

On a Metal build this exercises BOTH paths:
  - default device (GPU)      -> the fused Metal kernel
  - stream=mx.cpu             -> the op-level composition fallback
Both must match the oracle. Run from the worktree root with the venv python:
    .venv/bin/python test_spec_verify.py
"""
import sys
import mlx.core as mx


def oracle(draft, tgt_tokens):
    B, K = len(draft), len(draft[0])
    n_acc, committed = [], []
    for b in range(B):
        na = K
        for j in range(K):
            if draft[b][j] != tgt_tokens[b][j]:
                na = j
                break
        n_acc.append(na)
        committed.append(draft[b][:na] + [tgt_tokens[b][na]])
    return n_acc, committed


def onehot_logits(tgt_tokens, V, dtype):
    B, L = len(tgt_tokens), len(tgt_tokens[0])
    lg = [[[-10.0] * V for _ in range(L)] for _ in range(B)]
    for b in range(B):
        for j in range(L):
            lg[b][j][tgt_tokens[b][j]] = 10.0
    return mx.array(lg).astype(dtype)


CASES = [
    ("full-accept", [[5, 6, 7]], [[5, 6, 7, 9]]),
    ("mismatch@2", [[5, 6, 7]], [[5, 6, 1, 9]]),
    ("reject-all", [[5, 6, 7]], [[2, 6, 7, 9]]),
    ("K=1", [[3]], [[3, 4]]),
    ("batch2-K4", [[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 9, 4, 7], [1, 2, 3, 4, 8]]),
    ("batch3-K1", [[7], [7], [7]], [[7, 1], [2, 3], [7, 9]]),
]

PATHS = [("kernel(gpu)", None), ("fallback(cpu)", mx.cpu)]


def run():
    print(f"default_device={mx.default_device()}")
    V = 32
    allok = True
    for path_name, stream in PATHS:
        for dt in (mx.float32, mx.float16, mx.bfloat16):
            for name, draft, tgt in CASES:
                exp_n, exp_c = oracle(draft, tgt)
                lg = onehot_logits(tgt, V, dt)
                kw = {"stream": stream} if stream is not None else {}
                n, c = mx.fast.spec_decode_verify(
                    mx.array(draft, dtype=mx.int32), lg, **kw)
                mx.eval(n, c)
                n, c = n.tolist(), c.tolist()
                got_c = [c[b][: exp_n[b] + 1] for b in range(len(draft))]
                ok = (n == exp_n) and (got_c == exp_c)
                allok &= ok
                if not ok:
                    print(f"  [FAIL] {path_name} {str(dt):>16} {name}: "
                          f"got n={n} c={got_c}  exp n={exp_n} c={exp_c}")
        print(f"  [{'OK ' if allok else 'FAIL'}] path={path_name}: all cases x fp32/fp16/bf16")

    try:
        mx.fast.spec_decode_verify(mx.zeros((1, 3), dtype=mx.int32), mx.zeros((1, 3, 8)))
        print("  [FAIL] expected error on bad K+1 shape"); allok = False
    except Exception:
        print("  [OK ] rejects target_logits.shape[1] != K+1")

    print("ALL OK" if allok else "SOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(run())
