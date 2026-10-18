#!/usr/bin/env python3
"""Turn a Strata config written by setup.sh into the tuned "gamer 16 GB" config, plus a launcher next to it.

    python3 gamer-16gb/apply-tuning.py strata-IQ3_S.json [--experts-copy /fast/drive/experts.bin] [--no-turbo]
                                       [--ctx 131072] [--pool-workers N] [--name NAME]

Writes strata-NAME.json and run-NAME.sh in the Strata folder (NAME defaults to "<config name>-gamer-<ctx/1024>k").
Nothing is started. See gamer-16gb/README.md for what each setting does and what it was measured against.

Tuned for: one 16 GB GPU (measured on an RX 9060 XT), 16 GB of system RAM, the experts read from an NVMe SSD (a SATA
SSD works, slower), a 6-core CPU, Qwen3.8-Flash-Next GSQ-RCO IQ3_S at 128K context. KV stays int8 and the model files
are untouched. The turbo settings (on unless --no-turbo) change the output a little: measured against IQ3_S with every
output-changing setting off, they stay about half as far from it as IQ3_XXS is (README, "Quality").
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# flag -> value (None: a switch without a value). Replaced when the config has the flag, appended otherwise.
ARGS = {
    "--mmap-experts": None,        # experts stay in the model files; the OS page cache is the RAM tier (16 GB RAM)
    "--kv": "int8",                # int8 KV, as upstream validates it
    "--kv-grow": None,
    "--expert-cache": "auto",      # every byte of VRAM the model and KV leave goes to the GPU's expert slots
    "--vram-reserve-mib": "600",   # leaves ~260 MiB free in decode; 400 measured the same, raise it if the desktop pages
    "--prefill": "auto:32768",
    "--spec": "6",                 # MTP drafts
    "--spec-min-p": "0.4",         # 0.5 lost ~5% on long context once the page release (FILE_RELEASE) was on
    "--mtp-hnorm": "stream",
    "--adapt-every": "1",          # adaptive GPU tier: re-rank the experts every request ...
    "--adapt-swaps": "16",         # ... swapping at most 16 per step
    "--adapt-min-gain": "3",
    "--adapt-decay": "0.9",
    "--pcie-frac": "0.6",          # share of cache misses sent over PCIe to the GPU instead of computed on the CPU
}
DROP = {"--resident-experts": 0, "--resident-budget-gib": 1}   # 16 GB of RAM cannot hold a RAM tier: the page cache is it

# speed settings that leave the output as it is (or, VERIFY_DEFER_CPU and ROUTE_TAIL_SKIP, as upstream ships them)
ENV = {
    "STRATA_IO_PREFETCH": "1",             # the engine's I/O threads read missed experts ahead of the CPU pool
    "STRATA_LOOKAHEAD": "0",               # router look-ahead off: it doubles drive reads at ~0.7 precision (-10..-25%)
    "STRATA_ROUTE_TAIL_SKIP": "7",         # upstream's CUDA default, off on HIP unless set
    "STRATA_PROFILE_ROUTING_ONLY": "1",    # rank experts by routing alone (patch: not "resident first")
    "STRATA_VERIFY_TIMEOUT_S": "300",      # a drive under a page-fault storm can stall a verify window past 20 s
    "STRATA_WATCHDOG_S": "180",
    "STRATA_FILE_RELEASE": "1",            # page out the page-cache copy of experts the GPU holds (helper thread)
    "STRATA_FUSE_HEAD_GR": "1",
    "STRATA_FUSE_CONVAB": "1",
    "STRATA_VERIFY_DEFER_CPU": "1",        # upstream PRs #2689/#2690: the CPU experts overlap the GPU's next work
    "STRATA_KV_LEND": "1",                 # the KV reservation's unused tail lends its VRAM to expert slots
    "STRATA_DEFER_FUSE": "1",
    "STRATA_ROUTE_PUB_STORE": "1",
    "STRATA_HC_TSUM": "1",
    "STRATA_PREFILL_LEND_PCT": "95",
    "STRATA_DEFER_COLD_LAST": "1",         # inside each deferred pass, the experts still on the drive go last
}
AMD_ENV = {
    "GPU_PINNED_MIN_XFER_SIZE": "1048576",
    "STRATA_COPY_HIP_PRIO": "1",
    "STRATA_DOORBELL_STORE": "1",
    "STRATA_NO_PCIE_WAIT": "1",            # no empty PCIe wait and launch per layer
}
NO_AVX512_ENV = {"STRATA_Q2_BITPLANE": "1"}   # the MTP drafter's Q2_0 rows on AVX2 (changes the drafts' last bits only)
COPY_ENV = {                               # with the experts copy: the prompt reads them with O_DIRECT, both drives
    "STRATA_PROMPT_DIRECT": "1",
    "STRATA_PROMPT_DIRECT2": "1",
    "STRATA_REFILL_DIRECT": "1",
}
# output-changing (measured in README "Quality"); --no-turbo uses the milder cold-skip only
TURBO_ENV = {
    "STRATA_ROUTE_PRIOR": "0.5",           # among near-tied experts, prefer the ones the GPU already holds ...
    "STRATA_ROUTE_PRIOR_WARM": "0.5",      # ... or the page cache does: only experts that need a drive read lose
    "STRATA_COLD_SKIP_BELOW": "0.15",      # a decode miss with routing weight below 0.15 is skipped (and warmed)
    "STRATA_DEFER_COLD_LAYERS": "3",       # a cold CPU expert may join the residual up to 3 layers late
}
NO_TURBO_ENV = {"STRATA_COLD_SKIP_BELOW": "0.10"}


def set_flag(args, flag, val):
    if flag in args:
        i = args.index(flag)
        if val is not None:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                args[i + 1] = val
            else:
                args.insert(i + 1, val)
    else:
        args += [flag] + ([val] if val is not None else [])


def drop_flag(args, flag, nvals):
    while flag in args:
        i = args.index(flag)
        del args[i:i + 1 + nvals]


def has_avx512():
    try:
        return "avx512f" in Path("/proc/cpuinfo").read_text()
    except OSError:
        return True                        # unknown: leave the default kernel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the strata-*.json setup.sh wrote")
    ap.add_argument("--ctx", type=int, default=131072, help="max context (default 131072)")
    ap.add_argument("--experts-copy", help="a copy of the pack's experts.bin on a faster drive (copy-experts.sh)")
    ap.add_argument("--no-turbo", action="store_true",
                    help="leave out the output-changing turbo settings (slower, closer to the plain model)")
    ap.add_argument("--pool-workers", type=int, help="CPU expert workers (default: half the CPU threads, at least 2)")
    ap.add_argument("--name", help="output name (strata-NAME.json, run-NAME.sh)")
    o = ap.parse_args()

    src = Path(o.config).resolve()
    c = json.loads(src.read_text())
    args = c["args"]
    for flag, n in DROP.items():
        drop_flag(args, flag, n)
    for flag, val in ARGS.items():
        set_flag(args, flag, val)
    set_flag(args, "--max-context", str(o.ctx))
    workers = o.pool_workers or max(2, (os.cpu_count() or 6) // 2)
    set_flag(args, "--pool-workers", str(workers))   # 3 on 6 cores: the I/O, PLE and kernel threads need the rest

    env = c.setdefault("env", {})
    env.update(ENV)
    if c.get("backend") == "hip":
        env.update(AMD_ENV)
    if not has_avx512():
        env.update(NO_AVX512_ENV)
    for k in TURBO_ENV:
        env.pop(k, None)
    env.update(NO_TURBO_ENV if o.no_turbo else TURBO_ENV)
    if o.experts_copy:
        p = Path(o.experts_copy).resolve()
        if not p.is_file():
            sys.exit(f"{p}: no such file (make it with gamer-16gb/copy-experts.sh)")
        if p.stat().st_uid != os.getuid():
            print(f"warning: {p} is not owned by you. Linux mincore() then reports every page cached and the engine's "
                  "prioritized reads never run (measured: long context 14.3 vs 17.6 tok/s). chown it, or mount the "
                  "drive with uid=<you>.", file=sys.stderr)
        env["STRATA_EXPERTS_HEAD"] = str(p)
        env.update(COPY_ENV)

    v = c.get("vision")
    if isinstance(v, dict):                # setup --vision cpu: an on-demand CPU encoder costs no decode speed
        if not v.get("gpu"):
            v["elastic"] = True            # starts with the first picture, stops 120 s after the last new one
            v.setdefault("threads", os.cpu_count() or 6)
        v["max_tokens"] = 576              # the fewest image tokens that still read 14 px IDE text (300 misread it)
        if "--vision" not in args:
            args.append("--vision")

    name = o.name or f"{src.stem.removeprefix('strata-')}-gamer-{o.ctx // 1024}k"
    out = ROOT / f"strata-{name}.json"
    if "log" in c:
        c["log"] = str(ROOT / f"strata-{name}.log")
    out.write_text(json.dumps(c, indent=1) + "\n")
    shared = src.with_name(src.stem + ".shared-settings.json")
    if shared.exists():
        shutil.copyfile(shared, ROOT / f"strata-{name}.shared-settings.json")

    py = ROOT / ".venv" / "bin" / "python"
    run = ROOT / f"run-{name}.sh"
    run.write_text(f"""#!/bin/sh
# Strata, tuned for a 16 GB GPU + 16 GB RAM + NVMe (gamer-16gb/README.md). Written by gamer-16gb/apply-tuning.py.
# The server and engine run in their own cgroup scope with swap off, so their memory never goes to zram/swap
# (a swapped-out engine stalls for seconds). Without systemd it starts plainly.
cd "{ROOT}"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    exec systemd-run --user --scope -q -p MemorySwapMax=0 \\
        "{py}" "{ROOT}/serve/server.py" --engine strata --config "{out}" --port {c.get("port", 8080)}
fi
exec "{py}" "{ROOT}/serve/server.py" --engine strata --config "{out}" --port {c.get("port", 8080)}
""")
    run.chmod(0o755)
    print(f"wrote {out.name} and {run.name} (pool workers {workers}, context {o.ctx}, "
          + ("no turbo" if o.no_turbo else "turbo on")
          + (f", experts from {env['STRATA_EXPERTS_HEAD']}" if o.experts_copy else ", experts from the pack")
          + (", vision on demand" if isinstance(v, dict) else ", no vision") + ")")
    print(f"start it with: {run}")


if __name__ == "__main__":
    main()
