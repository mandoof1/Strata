# Strata on an average gaming PC: 16 GB VRAM, 16 GB RAM, NVMe + SATA

This branch is Strata 0.1.43.1 plus a set of engine patches and a tuned configuration for a normal gaming PC, the
kind that has a 16 GB graphics card but only 16 GB of system RAM. Upstream Strata assumes 48 GB of RAM or more for
the IQ2_XS model. With 16 GB, almost every expert has to come from the SSD, and the stock engine spends most of its
time waiting on page faults. The patches below make those reads cheap enough that Qwen3.8-Flash-Next at IQ2_XS runs
at 30+ tokens/s on this hardware.

This is a personal fork for people with a similar setup. It is not meant to be merged upstream.

## The measured PC

| Part | What |
|---|---|
| GPU | AMD Radeon RX 9060 XT, 16 GB (gfx1200), ROCm 7.2.4 |
| CPU | Intel Core i5-9400F, 6 cores, no AVX-512 |
| RAM | 16 GB |
| Drives | a 256 GB DRAM-less NVMe (about 1.5 GB/s random reads) and a 500 GB SATA SSD (Crucial MX500) |
| OS | CachyOS (Arch), Linux 7.2 |

## Results

Decode speed in tokens/s, greedy, measured with the same two request sets every time: "agent" is four short
agent-style prompts with 500-token answers, "long context" replays a real 40-51K-token chat with 400-token answers.

| Model and setup | Agent | Long context |
|---|---:|---:|
| Swift 1.5 IQ2_XS, 128K context, this branch | **31-32** | **19.0** |
| Swift 1.5 IQ2_XS, 64K context, this branch | 33.2 | 20.0 |
| Coder IQ1_M, 128K, stock 0.1.43.1 engine | 11.2 | 5.5 |
| Coder IQ1_M, 128K, this branch | 25.7 | 22.6 |
| Swift 1.5 IQ3_XXS, 128K, this branch (not recommended here) | 22.1 | 13.0 |

IQ3_XXS is 27% bigger per expert, so fewer experts fit in VRAM and more come from the drive. On 16 GB of RAM that
costs about a third of the speed, and nothing in this branch closes that gap (see "What did not help").

The tuning is about speed only. The model files are not modified, KV stays int8, and the one output-changing setting
(`STRATA_ROUTE_TAIL_SKIP=7`) is the value upstream already ships as the CUDA default.

## What the patches do

All of them are in `src/core/expert_source.cpp`, `src/program/generate.cpp`, `src/core/verify.cpp`,
`src/prefill/prefill.cpp` and `serve/server.py`, marked with `local` comments. Each has an environment switch, so
you can turn one off and compare. The same changes as one file, for applying to an upstream checkout of commit
`a0116b2` (0.1.43.1): `git apply gamer-16gb/strata-gamer-16gb-0.1.43.1.patch`.

1. **Prioritized drive reads (io-prio).** The experts the current layer needs get their own queue ahead of any
   read-ahead, and the CPU computes the cached experts first while the cold ones are still being read
   (`STRATA_IO_PRIO=0` and `STRATA_COLD_LAST=0` turn this off). Long context went from 5.5 to 12.6 tok/s with this,
   look-ahead off and the adaptive tier settings below.
2. **One big read per expert (io-populate).** An I/O thread queues the whole expert with `readahead` and then maps it
   with `MADV_POPULATE_READ`, so the CPU kernels never take a page fault on it and no 2 MB copy goes through the L3
   cache (`STRATA_IO_POPULATE=0` for the old pread path).
3. **Batched routing on every HIP card.** The verify window's router runs as one projection and one top-k on all AMD
   cards, not only gfx906 (`STRATA_ROUTE_WINDOW=0` for the per-token launches).
4. **Refill only from the page cache.** Opt-in (`STRATA_REFILL_CACHED_ONLY=1`); it lost on long context, so it stays
   off.
5. **Rank tie-break.** When the adaptive GPU tier has to evict one of several experts with the same usage, the one the
   profile ranks lowest goes (`STRATA_ADAPT_RANK_TIEBREAK=0` for the old order). `STRATA_PROFILE_ROUTING_ONLY=1`
   ranks by routing alone.
6. **Experts from a faster drive (split file).** `STRATA_EXPERTS_HEAD=/path/to/copy` maps experts.bin from a copy on
   another drive, so the pack can live on the big SATA SSD while the experts are read from the NVMe. The engine checks
   16 pages of the copy against experts.bin at every start and falls back to experts.bin if they differ. A copy of
   only the first N bytes works too.
7. **Read-ahead in 128 KiB steps.** The kernel silently clips one `readahead()` or `MADV_WILLNEED` call to the device's
   `read_ahead_kb` (256 KiB on most NVMe drives), so patch 2 only got the first 256 KiB of each 1.5-2 MB expert ahead
   and faulted in the rest synchronously. Issuing it in 128 KiB steps: a cold 1.5 MB expert 3.0 -> 1.8 ms, agent
   +9.5%, long context +16%.
8. **NVMe status line.** At start the server prints whether the experts are read from the fast copy and from which
   drive, or why not:

   ```
   [strata] NVMe copy: ON - all 33.02 GiB of experts are read from /mnt/win/strata-data/swift2-experts.bin (nvme0n1p3)
   [strata] NVMe copy: OFF - /mnt/win/... not used (cannot open it); all experts are read from .../experts.bin (sda2) - slower
   ```

Two more env settings from upstream that this branch's config uses: `STRATA_VERIFY_TIMEOUT_S=300` (a SATA SSD in a
page-fault storm can stall a window past the default 20 s without being stuck) and `STRATA_LOOKAHEAD=0`.

## Setup

1. **Clone this branch and run Strata's setup as usual.** The source differs from any release, so setup builds the
   engine from source (it needs the ROCm or CUDA toolchain; see the main README).

   ```bash
   git clone -b gamer-16gb https://github.com/mandoof1/Strata
   cd Strata
   ./setup.sh --setup --model IQ2_XS
   ```

   Setup writes a config such as `strata-IQ2_XS.json`. It also starts the server once; stop it before step 2.

   For **Swift 1.5** (UkisAI's fine-tune of the same model that thinks about half as long, the one measured above),
   download the two IQ2_XS shards from
   [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
   pack them, and point a copy of setup's config at the new pack and shard 1:

   ```bash
   STRATA_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
       --gguf /path/to/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
       --out /path/to/packs/swift-iq2_xs --experts-bin
   ```

   In the copied config, change `--pack`, `--native`, `--ple-gguf` (the shard that holds `per_layer_token_embd`,
   shard 1 for Swift) and `tokenizer` (`<pack>/tokenizer`).

2. **Put experts.bin on the NVMe (optional, recommended if the pack is on a SATA SSD).**

   ```bash
   gamer-16gb/copy-experts.sh /path/to/pack /path/on/nvme/experts.bin
   ```

   Run it as yourself. If the copy belongs to another user (for example root, or an NTFS partition mounted without
   `uid=`), Linux `mincore()` reports every page as cached, and the prioritized reads never run. That alone cost 19%
   on long context here.

3. **Write the tuned config and launcher.**

   ```bash
   python3 gamer-16gb/apply-tuning.py strata-IQ2_XS.json --experts-copy /path/on/nvme/experts.bin
   ./run-IQ2_XS-gamer-128k.sh
   ```

   `--ctx 65536` gives about 5% more speed (more VRAM left for expert slots) at half the context. The launcher runs
   the server in a systemd scope with swap off, so the engine's memory never goes to zram or swap.

## The settings, and why

| Setting | Value | Why |
|---|---|---|
| `--mmap-experts` | on | 16 GB of RAM cannot hold the experts; the page cache becomes the RAM tier. A static RAM tier or `--resident-budget-gib` measured no better (LRU is within 23% of optimal here). |
| `--pool-workers` | cores - 2 (4 on 6 cores) | leaves a core for the I/O, PLE and kernel threads that otherwise preempt the barrier-coupled pool: agent +22%, long context +7% |
| `--adapt-every 1 --adapt-swaps 8 --adapt-min-gain 3 --adapt-decay 0.9` | | the adaptive GPU tier follows what the conversation uses. Upstream's default (`--adapt-every 4`) was the slowest and crashed here |
| `--spec 5 --spec-min-p 0.5` | | MTP drafts; 4 and 6 measured within +-3% |
| `--pcie-frac 0.6` | | share of cache misses uploaded to the GPU instead of computed on the CPU |
| `--kv int8`, `--kv-grow`, `--expert-cache auto`, `--prefill auto` | | upstream defaults for this card |
| `STRATA_LOOKAHEAD` | 0 | the router look-ahead guesses the next layer's experts at ~0.7 precision and doubles the drive reads: -10 to -25% here |
| `STRATA_ROUTE_TAIL_SKIP` | 7 | upstream's CUDA default, off on HIP unless set |
| `STRATA_IO_PREFETCH` | 1 | the I/O threads of patches 1, 2 and 7 |

## What did not help (measured, so you do not have to)

- Router look-ahead on (`STRATA_LOOKAHEAD=1`): -10 to -25%.
- Reading part of every expert from the SATA SSD while the NVMe reads the rest. Raw throughput rises from 1.03 to
  1.53 GB/s, but the engine got 9-17% slower: decode only pulls ~0.7 GB/s and waits on each layer's slowest read,
  so a second, slower drive adds latency, not speed. Striping whole experts across the drives lost 2.5-4%.
- More I/O threads (16), the `none` I/O scheduler instead of kyber, submit-only reads: noise or worse.
- A learned expert profile from your own chats: no gain over the generic `data/expert-profile.bin`.
- `--resident-budget-gib`, a static RAM tier, `--prefill 2048`, `--spec 6`, `HSA_ENABLE_SDMA=0`, GFXOFF off,
  `vm.watermark_scale_factor` / `min_free_kbytes`: no gain.

## Tips

- Close the browser and anything else large while it runs: every GB of free RAM is page cache for the experts.
- Check `top` before comparing numbers. A runaway background process silently skewed a whole benchmark session here.
- The engine log line `io-prio: N demand reads` should show N > 0 per request. If it is 0, the prioritized reads are
  not running (usually the file-ownership problem from step 2).

## License

MIT, as upstream. Strata is by Niko1221 and the Strata contributors; the GSQ-RCO quantizations are by ISTA-DASLab.
