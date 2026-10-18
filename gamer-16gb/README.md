# Strata on an average gaming PC: 16 GB VRAM, 16 GB RAM, NVMe + SATA

This branch is Strata 0.1.44.1 (upstream `296b59d`) plus engine patches and a tuned configuration for a normal gaming
PC, the kind that has a 16 GB graphics card but only 16 GB of system RAM. Upstream Strata wants 62 GB of RAM for the
IQ3_S model. With 16 GB, almost every expert has to come from the SSD, and the stock engine spends most of its time
waiting on the drive. The patches below make those reads cheap and rarer, so Qwen3.8-Flash-Next **IQ3_S** (the best
quality size, "matches the full model") runs at **128K context with int8 KV at 27-39 tokens/s**, and it can read
pictures without losing decode speed.

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

Decode speed in tokens/s, greedy, 128K context, KV int8, measured 2026-10-18 with the same request sets every time:

- **Agent:** four short agent-style prompts, 500-token answers.
- **Long context:** a real 39-51K-token chat replayed, 400-token answers.
- **Agent session:** a real coding-agent session (pi), four consecutive turns at 82-87K tokens of context, 500-token
  answers. This is the closest to daily use.
- **+ screenshot:** the same session with a 1920x1080 screenshot pasted in, kept in all four turns.

| Model and setup | Agent | Long context | Agent session | + screenshot |
|---|---:|---:|---:|---:|
| **IQ3_S, this branch (default, turbo on)** | **38.6** | **34.7** | **27.3** | **26.4** |
| IQ3_S, this branch, `--no-turbo` | 27.3 | 13.7 | 16.7 | 13.6 |
| IQ3_XXS, this branch, `--no-turbo` | 37.1 | 23.5 | | |

Turbo numbers are three-run averages except the screenshot column (one run; a single run varies by about 10%). The
IQ3_XXS row is from 2026-10-17 with the same engine. Prompt reading: a 39K-token prompt reads at
about 430 tokens/s with the experts on the NVMe.

IQ3_S experts are 17% bigger than IQ3_XXS ones, so fewer fit in VRAM (4573 slots vs 5705) and more come from the
drive: without the turbo settings IQ3_S is 25-42% slower than IQ3_XXS. With them it is faster than IQ3_XXS and still
closer to the plain IQ3_S model than IQ3_XXS is (next section).

## Quality

The turbo settings change the output a little, so they were measured. Each setup reads the same fixed text (upstream's
`data/quant_quality/v1`: code, English, multilingual and agent segments, 4 x 1000 tokens, teacher-forced) and is
compared with IQ3_S running with every output-changing setting off. KL is the mean KL divergence over the reference's
64 most likely tokens (lower is closer); top-1 is how often both pick the same next token.

| Setup | KL | Top-1 | Perplexity |
|---|---:|---:|---:|
| IQ3_S, every output-changing setting off (the reference) | 0 | 100% | 3.220 |
| IQ3_S, `--no-turbo` | 0.075 | 92.4% | 3.212 |
| **IQ3_S, turbo (default)** | **0.109** | **89.9%** | 3.27-3.33 |
| IQ3_XXS, every output-changing setting off | 0.165 | 88.2% | 3.263 |
| IQ3_XXS, `--no-turbo` | 0.173 | 86.9% | 3.269 |

So the turbo moves IQ3_S about two thirds as far as switching to IQ3_XXS would, and it is faster than IQ3_XXS. The
perplexity of 4000 tokens moves by about 0.05 from run to run with settings that depend on drive timing; KL is the
stable number. `ROUTE_PRIOR=1.0` (48.8 / 41.4 / 30.6 tokens/s) costs KL 0.185, more than IQ3_XXS, so it is not used.

## Pictures (vision)

`setup.sh --vision cpu` installs the image encoder; `apply-tuning.py` makes it start with the first picture and stop
120 s after the last new one. Measured on the agent session above:

- **Decode speed with pictures:** unchanged (26.4 vs 27.3 tokens/s, within run-to-run noise).
- **A new picture:** about 20 s once (the encoder starts in 2-4 s, a 1080p screenshot encodes in ~8 s on 6 CPU threads,
  then 576 image tokens are read). A picture already in the conversation costs nothing: agents send their old pictures
  again with every turn, and those are served from the embedding cache without starting the encoder.
- **RAM:** the running encoder takes about 1 GB, given back when it stops.
- **576 image tokens** are the fewest that still read 14 px IDE text exactly (300 misread it; 1024 only adds encode
  time).
- **GPU encoder** (`tools/vision` with `-DSTRATA_VISION_HIP=ON`): 0.6 s per picture, but it keeps its VRAM all the
  time (the HIP engine has no `--vram-elastic`) and decode loses 12%. Not used.

## What the patches do

Every change has an environment switch (most are off by default and turned on by `apply-tuning.py`), so you can turn
one off and compare. To see them all as one diff against upstream: `git diff 296b59d gamer-16gb -- src include serve`.

**Drive reads (from the first version of this branch):**

1. **Prioritized drive reads.** The experts the current layer needs get their own queue ahead of any read-ahead, and
   the CPU computes the cached experts first while the cold ones are still being read (`STRATA_IO_PRIO=0`,
   `STRATA_COLD_LAST=0` turn this off).
2. **One big read per expert.** An I/O thread reads the whole expert with `readahead` and maps it with
   `MADV_POPULATE_READ`, in 128 KiB steps (the kernel silently clips one call to the drive's `read_ahead_kb`), so the
   CPU kernels never take a page fault on it (`STRATA_IO_POPULATE=0`).
3. **Batched routing on every HIP card** (`STRATA_ROUTE_WINDOW=0` for the per-token launches).
4. **Rank tie-break** in the adaptive GPU tier (`STRATA_ADAPT_RANK_TIEBREAK=0`); `STRATA_PROFILE_ROUTING_ONLY=1` ranks
   by routing alone.
5. **Experts from a faster drive.** `STRATA_EXPERTS_HEAD=/path/to/copy` maps experts.bin from a copy on another drive
   (16 pages are checked against experts.bin at every start; a mismatch falls back to experts.bin). At start the server
   prints `NVMe copy: ON - ... (nvme0n1p3)` or why it is off.

**Memory and VRAM:**

6. **Page release** (`STRATA_FILE_RELEASE=1`): once the GPU holds an expert, its page-cache copy is paged out on a helper
   thread, so 16 GB of RAM caches more of the experts the GPU does not hold. Agent +11%.
7. **KV lending** (`STRATA_KV_LEND=1`, exact): the part of the 128K KV reservation the conversation has not reached yet
   holds extra expert slots until it is needed (1075 more slots with IQ3_XXS on this card). Plus a static VRAM
   compaction at load (exact).

**Prompt reading:**

8. **Direct prompt reader** (`STRATA_PROMPT_DIRECT=1`): a long prompt reads its experts with `O_DIRECT` on 8 threads
   from the NVMe copy instead of through page faults; `STRATA_PROMPT_DIRECT2=1` also reads from the pack's experts.bin
   on the SATA SSD as a second drive, and `STRATA_REFILL_DIRECT=1` does the same for the GPU refill after the prompt.
   With these and `--prefill auto:32768`, four long-chat prompts took 130 s instead of 211 s.

**Decode kernels and scheduling (exact unless noted):**

9. Upstream PRs merged here before upstream did: #2492 (gfx12 matvec grid padding), #2689/#2690
   (`STRATA_VERIFY_DEFER_CPU=1`: the CPU experts overlap the GPU's next work; upstream measured ~97% top-1 agreement),
   #2215 (bit-exact faster one-token AVX2 rows for IQ3_S / IQ3_XXS / IQ2_S, default on), #2751 and #1415/#1544
   (opt-in, off here).
10. The HIP decode kernel bundle from the android-imdad fork (withdrawn upstream PR #2231), `STRATA_DEFER_FUSE=1`
    (fused deferred-CPU waits), `STRATA_ROUTE_PUB_STORE=1` (top-k and doorbell in one launch), `STRATA_HC_TSUM=1`
    (HC Q8 reads 3-25% faster), a measured gfx1200 table for the interleaved GEMV, `STRATA_NO_PCIE_WAIT=1` (no empty
    PCIe wait and launch per layer).

**Cold misses (output-changing, opt-in; the "turbo"):**

11. `STRATA_DEFER_COLD_LAST=1` (same output): inside each deferred CPU pass, the experts still on the drive go last.
12. `STRATA_COLD_SKIP_BELOW=w`: in decode, an expert that missed both VRAM and the page cache and has a routing weight
    below `w` is skipped and read in the background for next time (0.10 without turbo, 0.15 with).
13. `STRATA_DEFER_COLD_LAYERS=k`: a cold CPU expert may join the residual up to `k` layers late instead of stalling the
    layer (3 with turbo).
14. `STRATA_ROUTE_PRIOR=l` (upstream): among near-tied experts, the router prefers the ones the GPU holds. This branch
    adds `STRATA_ROUTE_PRIOR_WARM=w`: experts in the page cache get a bonus too, so only experts that need a drive read
    are displaced. At 0.5 / 0.5 this kept the fidelity of the prior alone (KL 0.104) and was faster on long context.
    `STRATA_ROUTE_PRIOR_KEEP=k` never displaces the router's own top k (measured, not used).

**Server:**

15. A CPU image encoder starts on demand without `vram_elastic` (which the HIP engine refuses), takes no VRAM from the
    expert cache, and is not started for pictures already in the embedding cache.

**Quality measurement:** `STRATA_LOGPOS_DECODE=1` makes teacher-forced scoring windows count as decode windows, so the
decode-only switches above act there too (used for the Quality table).

## Setup

1. **Clone this branch and run Strata's setup.** The source differs from any release, so setup builds the engine from
   source (it needs ROCm or CUDA; see the main README).

   ```bash
   git clone -b gamer-16gb https://github.com/mandoof1/Strata
   cd Strata
   ./setup.sh --setup --model IQ3_S --context 131072 --vision cpu
   ```

   Setup warns that IQ3_S needs about 62 GB of RAM: answer yes (or add `--yes`); step 3 switches the config to reading
   the experts through the page cache. Setup writes a config such as `strata-IQ3_S.json` and starts the server once:
   stop it before step 2. (The measured engine was built with `-DSTRATA_HIP_MMVQ_IL` in `CMAKE_HIP_FLAGS`; an A/B
   showed no speed difference, so setup's plain build is fine.)

2. **Put experts.bin on the NVMe (recommended when the pack is on a SATA SSD).** It is 50.3 GB for IQ3_S.

   ```bash
   gamer-16gb/copy-experts.sh /path/to/pack /path/on/nvme/experts.bin
   ```

   Run it as yourself. If the copy belongs to another user (for example root, or an NTFS partition mounted without
   `uid=`), Linux `mincore()` reports every page as cached and the prioritized reads never run. That alone cost 19% on
   long context here.

3. **Write the tuned config and launcher, and start it.**

   ```bash
   python3 gamer-16gb/apply-tuning.py strata-IQ3_S.json --experts-copy /path/on/nvme/experts.bin
   ./run-IQ3_S-gamer-128k.sh
   ```

   Add `--no-turbo` for the closest-to-the-model settings (slower, see Results). The launcher runs the server in a
   systemd scope with swap off, so the engine's memory never goes to zram or swap.

4. **Pictures.** Send them as OpenAI `image_url` parts (or Anthropic image blocks). In a coding agent, mark the model
   as accepting images: in pi, set `"input": ["text", "image"]` on the model in `~/.pi/agent/models.json`, then paste a
   screenshot with Ctrl+V or let the agent open an image file.

## The settings, and why

| Setting | Value | Why |
|---|---|---|
| `--mmap-experts` | on | 16 GB of RAM cannot hold the experts; the page cache becomes the RAM tier. A static RAM tier or `--resident-budget-gib` measured no better. |
| `--pool-workers` | half the CPU threads (3 on 6 cores) | leaves cores for the I/O, PLE and kernel threads that otherwise preempt the barrier-coupled pool |
| `--adapt-every 1 --adapt-swaps 16 --adapt-min-gain 3 --adapt-decay 0.9` | | the adaptive GPU tier follows what the conversation uses |
| `--spec 6 --spec-min-p 0.4` | | MTP drafts; min-p 0.5 lost ~5% on long context with the page release on |
| `--vram-reserve-mib 600`, `--prefill auto:32768`, `--mtp-hnorm stream`, `--pcie-frac 0.6` | | leave ~260 MiB VRAM free in decode; prompt chunking; MTP norm streaming; share of misses uploaded to the GPU |
| `--kv int8`, `--kv-grow`, `--expert-cache auto` | | upstream defaults for this card |
| `STRATA_LOOKAHEAD` | 0 | the router look-ahead doubles the drive reads at ~0.7 precision: -10 to -25% here |
| `STRATA_ROUTE_TAIL_SKIP` | 7 | upstream's CUDA default, off on HIP unless set |
| `STRATA_IO_PREFETCH` | 1 | the I/O threads of patches 1 and 2 |
| `STRATA_VERIFY_TIMEOUT_S` | 300 | a drive in a page-fault storm can stall a window past the default 20 s without being stuck |
| `STRATA_Q2_BITPLANE` | 1 on CPUs without AVX-512 | upstream's AVX2 rows for the Q2_0 MTP drafter |
| `STRATA_COPY_HIP_PRIO`, `STRATA_DOORBELL_STORE`, `STRATA_FUSE_HEAD_GR`, `STRATA_FUSE_CONVAB`, `STRATA_PREFILL_LEND_PCT=95` | | upstream opt-ins that measured faster here |
| turbo: `STRATA_ROUTE_PRIOR=0.5 STRATA_ROUTE_PRIOR_WARM=0.5 STRATA_COLD_SKIP_BELOW=0.15 STRATA_DEFER_COLD_LAYERS=3` | | the Results / Quality trade above; `--no-turbo` keeps only `STRATA_COLD_SKIP_BELOW=0.10` |

## What did not help (measured, so you do not have to)

On IQ3_S (2026-10-18), against the same settings without the change:

- A learned expert profile from your own chats: -9% agent (the adaptive tier re-learns anyway).
- `STRATA_VERIFY_DEFER_CPU_BELOW=1`: slower, and KL 0.176 (as far from the model as IQ3_XXS).
- The exact CPU kernels `STRATA_IQ256_GATHER=1` + `STRATA_IQ3S_MT1=1`: -9% agent, -13% agent session.
- `--mtp-q4 all` (smaller drafter, more expert slots): -14% agent.
- `--vram-reserve-mib 400`: no change. `STRATA_ROUTE_TAIL_SKIP=6`: faster on short prompts, slower on the agent session.
- `STRATA_COLD_SKIP_BELOW=0.20` (instead of 0.15): no faster, more KL. `STRATA_ROUTE_PRIOR_WARM=1.0`: slower.
- `STRATA_ROUTE_PRIOR=1.0`, alone or with `STRATA_ROUTE_PRIOR_KEEP=2`: the fastest (agent session 28-31 tokens/s) but
  KL 0.18-0.19, more than IQ3_XXS.
- The GPU image encoder: -12% decode (above).

From the earlier versions of this branch (0.1.43.1 / 0.1.44.1, Swift IQ2_XS and IQ3_XXS):

- Router look-ahead on, on the CPU or the GPU: -10 to -25%.
- Reading part of every expert from the SATA SSD while the NVMe reads the rest: -9 to -17% (decode waits on each
  layer's slowest read). Striping whole experts across the drives: -2.5 to -4%.
- More I/O threads, the `none` I/O scheduler, submit-only reads, `--resident-budget-gib`, a static RAM tier,
  `HSA_ENABLE_SDMA=0`, GFXOFF off, `vm.watermark_scale_factor` / `min_free_kbytes`, MGLRU off, `read_ahead_kb` 512:
  noise or worse.

## Tips

- Close the browser and anything else large while it runs: every GB of free RAM is page cache for the experts.
- Keep `/tmp` small if it is a tmpfs: 4.6 GB of leftovers in `/tmp` (RAM, swapped to zram) cost 10-25% here.
- Check `top` before comparing numbers. A runaway background process silently skewed a whole benchmark session here.
- The engine log line `io-prio: N demand reads` should show N > 0 per request. If it is 0, the prioritized reads are
  not running (usually the file-ownership problem from step 2).

## Earlier versions of this branch

The first version (Strata 0.1.43.1, 2026-10-15) ran Swift 1.5 IQ2_XS at 31-32 tokens/s (agent) and 19.0 (long
context) on the same PC. Its patches (1-5 above) are part of this version.

## License

MIT, as upstream. Strata is by Niko1221 and the Strata contributors; the GSQ-RCO quantizations are by ISTA-DASLab.
