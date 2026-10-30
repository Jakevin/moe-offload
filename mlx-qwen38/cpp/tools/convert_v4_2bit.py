#!/usr/bin/env python3
"""Lossless converter from K50 v3 (ternary "tern" records) to K50 v4 ("2t" 2-bit affine).

Converts ONLY kind=="tern" expert records (5 base-3 trits/byte + bf16 scale) into the
exact runtime "2t" MLX affine 2-bit representation:
  - codes: uint32 packed 2-bit codes (16 trits per uint32 word, each trit in {-1,0,+1} mapped to {0,1,2})
  - scales: unchanged bfloat16 scales (group_size=64), symlinked directly to v3/Lxx/scales.bf16
  - biases: bfloat16 bias = -scale (IEEE-754 bf16 sign-bit XOR 0x8000), stored in v4/Lxx/bias.bf16
  - index.json: kind changed from "tern" -> "2t" (bits=2), preserving separation from original "2"-bit experts

All 2/3/4-bit expert records and all dense records in codes.bin / bias.bf16 are copied
byte-for-byte without re-quantization. All non-expert files (skeleton.safetensors,
nonexpert.safetensors, ple/, tokenizer, configs) are symlinked to v3 without modifying v3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GROUP_SIZE = 64


def build_base3_lut() -> np.ndarray:
    """Build (256, 5) uint32 lookup table mapping each base-3 byte to 5 unsigned 2-bit codes in {0, 1, 2}."""
    lut = np.zeros((256, 5), dtype=np.uint32)
    for b in range(256):
        v = b
        for k in range(5):
            lut[b, k] = v % 3
            v //= 3
    return lut


BASE3_LUT = build_base3_lut()
SHIFTS_16 = (np.arange(16, dtype=np.uint32) * 2)[None, None, :]


def unpack_base3_to_affine2_u32(raw_codes_u8: np.ndarray, n: int, k: int, row_bytes: int) -> np.ndarray:
    """Unpack (n, row_bytes) base-3 bytes into (n, k // 16) uint32 MLX affine 2-bit words."""
    if raw_codes_u8.size != n * row_bytes:
        raise ValueError(f"Expected {n * row_bytes} bytes, got {raw_codes_u8.size}")
    if k % 16 != 0 or k % 5 != 0 or row_bytes != k // 5:
        raise ValueError(f"Invalid ternary shape=({n}, {k}) row_bytes={row_bytes}")
    raw_2d = raw_codes_u8.reshape(n, row_bytes)
    # (n, k // 5, 5) -> (n, k // 16, 16)
    trits = BASE3_LUT[raw_2d].reshape(n, k // 16, 16)
    packed_u32 = np.bitwise_or.reduce(trits << SHIFTS_16, axis=-1).astype(np.uint32)
    return np.ascontiguousarray(packed_u32)


def negate_bf16_bytes(raw_scales_u8: np.ndarray) -> np.ndarray:
    """Negate bfloat16 tensor bytes via sign-bit XOR 0x8000 (exact match to -scales.astype(bfloat16))."""
    u16 = np.frombuffer(raw_scales_u8, dtype=np.uint16)
    neg_u16 = (u16 ^ np.uint16(0x8000)).astype(np.uint16)
    return neg_u16.view(np.uint8)


def setup_top_level_symlinks(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_dir.iterdir()):
        name = entry.name
        if name.startswith("L") and len(name) == 3 and name[1:].isdigit():
            continue
        if name == "__pycache__":
            continue
        dst_path = dst_dir / name
        if name == "EXPORT.json":
            export_data = json.loads(entry.read_text(encoding="utf-8"))
            export_data["version"] = "k50-v4-2bit"
            export_data["v3_source"] = str(src_dir)
            export_data["ternary_v4"] = (
                "kind='tern' experts pre-converted losslessly to kind='2t' "
                "(MLX uint32 2-bit affine codes, group_size=64, bf16 scale, bf16 bias=-scale)"
            )
            dst_path.write_text(json.dumps(export_data, indent=2) + "\n", encoding="utf-8")
            continue
        if dst_path.is_symlink() or dst_path.exists():
            continue
        os.symlink(entry.resolve(), dst_path)


def convert_layer(src_dir: Path, dst_dir: Path, layer_idx: int) -> dict:
    t0 = time.perf_counter()
    lname = f"L{layer_idx:02d}"
    src_l = src_dir / lname
    dst_l = dst_dir / lname
    dst_l.mkdir(parents=True, exist_ok=True)

    idx = json.loads((src_l / "index.json").read_text(encoding="utf-8"))

    # Symlink scales.bf16 and done.json directly to v3 (offsets and bytes are 100% unchanged)
    for sym_name in ("scales.bf16", "done.json"):
        src_f = src_l / sym_name
        dst_f = dst_l / sym_name
        if src_f.exists() and not (dst_f.is_symlink() or dst_f.exists()):
            os.symlink(src_f.resolve(), dst_f)

    tmp_codes = dst_l / "codes.bin.tmp"
    tmp_bias = dst_l / "bias.bf16.tmp"
    final_codes = dst_l / "codes.bin"
    final_bias = dst_l / "bias.bf16"

    code_off = 0
    bias_off = 0
    n_tern_converted = 0
    n_other_copied = 0
    n_dense_copied = 0

    with (
        open(src_l / "codes.bin", "rb") as f_src_codes,
        open(src_l / "scales.bf16", "rb") as f_src_scales,
        open(src_l / "bias.bf16", "rb") as f_src_bias,
        open(tmp_codes, "wb") as f_dst_codes,
        open(tmp_bias, "wb") as f_dst_bias,
    ):
        for proj in PROJECTIONS:
            rows = idx["experts"][proj]
            for rec in rows:
                n, k = (int(v) for v in rec["shape"])
                old_c_off = int(rec["code_off"])
                old_c_len = int(rec["code_nbytes"])
                f_src_codes.seek(old_c_off)
                raw_c = np.frombuffer(f_src_codes.read(old_c_len), dtype=np.uint8)
                if raw_c.size != old_c_len:
                    raise RuntimeError(f"{lname} {proj} short read on codes.bin")

                if rec["kind"] == "tern":
                    row_bytes = int(rec.get("row_bytes", k // 5))
                    packed_u32 = unpack_base3_to_affine2_u32(raw_c, n, k, row_bytes)
                    code_bytes = packed_u32.tobytes()

                    s_off = int(rec["scale_off"])
                    s_len = int(rec["scale_nbytes"])
                    f_src_scales.seek(s_off)
                    raw_s = np.frombuffer(f_src_scales.read(s_len), dtype=np.uint8)
                    if raw_s.size != s_len:
                        raise RuntimeError(f"{lname} {proj} short read on scales.bf16")
                    bias_bytes = negate_bf16_bytes(raw_s).tobytes()

                    f_dst_codes.write(code_bytes)
                    f_dst_bias.write(bias_bytes)

                    rec["kind"] = "2t"
                    rec["bits"] = 2
                    rec["row_bytes"] = (k // 16) * 4
                    rec["code_off"] = code_off
                    rec["code_nbytes"] = len(code_bytes)
                    rec["bias_off"] = bias_off
                    rec["bias_nbytes"] = len(bias_bytes)

                    code_off += len(code_bytes)
                    bias_off += len(bias_bytes)
                    n_tern_converted += 1
                else:
                    old_b_off = int(rec["bias_off"])
                    old_b_len = int(rec["bias_nbytes"])
                    f_src_bias.seek(old_b_off)
                    bias_bytes = f_src_bias.read(old_b_len)
                    if len(bias_bytes) != old_b_len:
                        raise RuntimeError(f"{lname} {proj} short read on bias.bf16")

                    f_dst_codes.write(raw_c.tobytes())
                    f_dst_bias.write(bias_bytes)

                    rec["code_off"] = code_off
                    rec["bias_off"] = bias_off
                    code_off += old_c_len
                    bias_off += old_b_len
                    n_other_copied += 1

        # Dense projections (attention, shared_expert)
        for dkey, rec in idx["dense"].items():
            if rec["kind"] == "tern":
                raise RuntimeError(f"Unexpected tern in dense record {lname}:{dkey}")
            old_c_off = int(rec["code_off"])
            old_c_len = int(rec["code_nbytes"])
            f_src_codes.seek(old_c_off)
            code_bytes = f_src_codes.read(old_c_len)
            if len(code_bytes) != old_c_len:
                raise RuntimeError(f"{lname} dense {dkey} short read on codes.bin")

            old_b_off = int(rec["bias_off"])
            old_b_len = int(rec["bias_nbytes"])
            f_src_bias.seek(old_b_off)
            bias_bytes = f_src_bias.read(old_b_len)
            if len(bias_bytes) != old_b_len:
                raise RuntimeError(f"{lname} dense {dkey} short read on bias.bf16")

            f_dst_codes.write(code_bytes)
            f_dst_bias.write(bias_bytes)

            rec["code_off"] = code_off
            rec["bias_off"] = bias_off
            code_off += old_c_len
            bias_off += old_b_len
            n_dense_copied += 1

    os.replace(tmp_codes, final_codes)
    os.replace(tmp_bias, final_bias)

    idx["nbytes"] = code_off
    (dst_l / "index.json").write_text(json.dumps(idx) + "\n", encoding="utf-8")

    dt = time.perf_counter() - t0
    return {
        "layer": layer_idx,
        "tern_converted": n_tern_converted,
        "other_copied": n_other_copied,
        "dense_copied": n_dense_copied,
        "codes_bytes": code_off,
        "bias_bytes": bias_off,
        "elapsed_s": dt,
    }


def verify_layer_against_v3_runtime(src_dir: Path, dst_dir: Path, layer_idx: int) -> dict:
    """Verify every expert & dense record in v4/L{layer_idx:02d} is bitwise identical to v3 runtime."""
    speed_dir = Path(os.environ.get("QWEN38_V3_RUNTIME_DIR", "."))  # dir containing the v3 Python runtime.py (only needed for --verify)
    if str(speed_dir) not in sys.path:
        sys.path.insert(0, str(speed_dir))
    import runtime as RT  # type: ignore
    import mlx.core as mx

    lname = f"L{layer_idx:02d}"
    pack_v3 = RT.ExpertPack(src_dir / lname)
    pack_v4 = RT.ExpertPack(dst_dir / lname)

    checked_tern = 0
    checked_other = 0
    checked_dense = 0

    for proj in PROJECTIONS:
        rows_v3 = pack_v3.index["experts"][proj]
        rows_v4 = pack_v4.index["experts"][proj]
        if len(rows_v3) != len(rows_v4):
            raise RuntimeError(f"Record count mismatch in {lname} {proj}")
        for r3, r4 in zip(rows_v3, rows_v4):
            assert int(r3["expert"]) == int(r4["expert"])
            v3_val = pack_v3.materialize_record(r3)
            if v3_val[0] == "tern":
                v3_val = RT.tern_to_affine2(v3_val)
                checked_tern += 1
            else:
                checked_other += 1

            v4_val = pack_v4.materialize_record(r4)
            kind3, c3, s3, b3, k3 = v3_val
            kind4, c4, s4, b4, k4 = v4_val
            mx.eval(c3, s3, b3, c4, s4, b4)

            if kind3 != kind4 or k3 != k4:
                raise RuntimeError(f"Metadata mismatch at {lname} {proj} e={r3['expert']}: {kind3}/{k3} vs {kind4}/{k4}")
            if bytes(memoryview(c3)) != bytes(memoryview(c4)):
                raise RuntimeError(f"Codes bitwise mismatch at {lname} {proj} e={r3['expert']}")
            if bytes(memoryview(s3)) != bytes(memoryview(s4)):
                raise RuntimeError(f"Scales bitwise mismatch at {lname} {proj} e={r3['expert']}")
            if bytes(memoryview(b3)) != bytes(memoryview(b4)):
                raise RuntimeError(f"Biases bitwise mismatch at {lname} {proj} e={r3['expert']}")

    for dkey in pack_v3.index["dense"]:
        r3 = pack_v3.index["dense"][dkey]
        r4 = pack_v4.index["dense"][dkey]
        _, c3, s3, b3, k3 = pack_v3.materialize_record(r3)
        _, c4, s4, b4, k4 = pack_v4.materialize_record(r4)
        mx.eval(c3, s3, b3, c4, s4, b4)
        if (
            k3 != k4
            or bytes(memoryview(c3)) != bytes(memoryview(c4))
            or bytes(memoryview(s3)) != bytes(memoryview(s4))
            or bytes(memoryview(b3)) != bytes(memoryview(b4))
        ):
            raise RuntimeError(f"Dense bitwise mismatch at {lname} {dkey}")
        checked_dense += 1

    pack_v3.close()
    pack_v4.close()
    return {
        "layer": layer_idx,
        "checked_tern": checked_tern,
        "checked_other": checked_other,
        "checked_dense": checked_dense,
        "all_bitwise_identical": True,
    }


def parse_layers_spec(spec: str) -> list[int]:
    if not spec or spec.strip().lower() == "all":
        return list(range(48))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert K50 v3 tern experts to K50 v4 2-bit affine ('2t')")
    parser.add_argument("--src", type=Path, required=True, help="K50 v3 model dir")
    parser.add_argument("--dst", type=Path, required=True, help="output K50 v4 (2t) model dir")
    parser.add_argument("--layers", type=str, default="all", help="Comma/range list of layers, e.g. '4' or '0-47' or 'all'")
    parser.add_argument("--verify", action="store_true", help="Verify converted layer(s) bitwise against v3 runtime")
    parser.add_argument("--skip-existing", action="store_true", help="Skip layers whose index.json already has no 'tern'")
    args = parser.parse_args()

    setup_top_level_symlinks(args.src, args.dst)
    layers = parse_layers_spec(args.layers)

    total_tern = 0
    total_other = 0
    t_all0 = time.perf_counter()

    for l_idx in layers:
        dst_idx_path = args.dst / f"L{l_idx:02d}" / "index.json"
        if args.skip_existing and dst_idx_path.is_file():
            txt = dst_idx_path.read_text(encoding="utf-8")
            if '"tern"' not in txt:
                print(f"[convert_v4] L{l_idx:02d} already converted, skipping", flush=True)
                if args.verify:
                    v_res = verify_layer_against_v3_runtime(args.src, args.dst, l_idx)
                    print(f"[verify_v4]  L{l_idx:02d} PASS: {v_res}", flush=True)
                continue

        res = convert_layer(args.src, args.dst, l_idx)
        total_tern += res["tern_converted"]
        total_other += res["other_copied"]
        print(
            f"[convert_v4] L{l_idx:02d} done in {res['elapsed_s']:.2f}s: "
            f"tern->2t={res['tern_converted']}, other={res['other_copied']}, "
            f"dense={res['dense_copied']}, codes={res['codes_bytes']/1e6:.1f}MB, "
            f"bias={res['bias_bytes']/1e6:.1f}MB",
            flush=True,
        )
        if args.verify:
            v_res = verify_layer_against_v3_runtime(args.src, args.dst, l_idx)
            print(f"[verify_v4]  L{l_idx:02d} PASS: {v_res}", flush=True)

    print(
        f"[convert_v4] Finished {len(layers)} layer(s) in {time.perf_counter() - t_all0:.2f}s "
        f"(total tern->2t={total_tern}, other={total_other})",
        flush=True,
    )


if __name__ == "__main__":
    main()
