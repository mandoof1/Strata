#!/usr/bin/env python3
"""Turn a Strata config written by setup.sh into the tuned "gamer 16 GB" config, plus a launcher next to it.

    python3 gamer-16gb/apply-tuning.py strata-IQ2_XS.json [--ctx 131072] [--experts-copy /fast/drive/experts.bin]
                                       [--pool-workers N] [--name NAME]

Writes strata-NAME.json and run-NAME.sh in the Strata folder (NAME defaults to "<config name>-gamer-<ctx/1024>k").
Nothing is started. See gamer-16gb/README.md for what each setting does and what it was measured against.

Tuned for: one 16 GB GPU (measured on an RX 9060 XT), 16 GB of system RAM, the experts read from an NVMe SSD (a SATA
SSD works, slower), a 6-core CPU. The settings are speed settings only: KV stays int8, the quantization is untouched.
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
    "--kv-grow": None,             # the KV grows with the conversation instead of being reserved up front
    "--expert-cache": "auto",      # every byte of VRAM the model and KV leave goes to the GPU's expert slots
    "--prefill": "auto",
    "--spec": "5",                 # MTP draft length (4 and 6 measured within +-3%)
    "--spec-min-p": "0.5",
    "--adapt-every": "1",          # adaptive GPU tier: re-rank the experts every request ...
    "--adapt-swaps": "8",          # ... swapping at most 8 per step (4: -3% long context, 16: worse both)
    "--adapt-min-gain": "3",
    "--adapt-decay": "0.9",
    "--pcie-frac": "0.6",          # share of cache misses sent over PCIe to the GPU instead of computed on the CPU
}
ENV = {
    "STRATA_IO_PREFETCH": "1",             # the engine's I/O threads read missed experts ahead of the CPU pool
    "STRATA_LOOKAHEAD": "0",               # router look-ahead off: it doubles drive reads at ~0.7 precision (-10..-25%)
    "STRATA_ROUTE_TAIL_SKIP": "7",         # upstream's CUDA default, off on HIP unless set
    "STRATA_PROFILE_ROUTING_ONLY": "1",    # rank experts by routing alone (patch: not "resident first")
    "STRATA_VERIFY_TIMEOUT_S": "300",      # a drive under a page-fault storm can stall a verify window past 20 s
    "STRATA_WATCHDOG_S": "180",
}
AMD_ENV = {"GPU_PINNED_MIN_XFER_SIZE": "1048576"}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the strata-*.json setup.sh wrote")
    ap.add_argument("--ctx", type=int, default=131072, help="max context (131072; 65536 is ~5%% faster)")
    ap.add_argument("--experts-copy", help="a copy of the pack's experts.bin on a faster drive (copy-experts.sh)")
    ap.add_argument("--pool-workers", type=int, help="CPU expert workers (default: CPU cores - 2, at least 2)")
    ap.add_argument("--name", help="output name (strata-NAME.json, run-NAME.sh)")
    o = ap.parse_args()

    src = Path(o.config).resolve()
    c = json.loads(src.read_text())
    args = c["args"]
    for flag, val in ARGS.items():
        set_flag(args, flag, val)
    set_flag(args, "--max-context", str(o.ctx))
    workers = o.pool_workers or max(2, (os.cpu_count() or 6) - 2)
    set_flag(args, "--pool-workers", str(workers))   # one core free for the I/O / PLE threads: +22% agent speed on 6 cores

    env = c.setdefault("env", {})
    env.update(ENV)
    if c.get("backend") == "hip":
        env.update(AMD_ENV)
    if o.experts_copy:
        p = Path(o.experts_copy).resolve()
        if not p.is_file():
            sys.exit(f"{p}: no such file (make it with gamer-16gb/copy-experts.sh)")
        if p.stat().st_uid != os.getuid():
            print(f"warning: {p} is not owned by you. Linux mincore() then reports every page cached and the engine's "
                  "prioritized reads never run (measured: long context 14.3 vs 17.6 tok/s). chown it, or mount the "
                  "drive with uid=<you>.", file=sys.stderr)
        env["STRATA_EXPERTS_HEAD"] = str(p)

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
    print(f"wrote {out.name} and {run.name} (pool workers {workers}, context {o.ctx}"
          + (f", experts from {env['STRATA_EXPERTS_HEAD']}" if o.experts_copy else ", experts from the pack") + ")")
    print(f"start it with: {run}")


if __name__ == "__main__":
    main()
