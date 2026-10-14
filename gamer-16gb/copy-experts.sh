#!/bin/bash
# copy-experts.sh PACK_DIR DEST_FILE
#
# Copies a pack's experts.bin (the routed experts, ~35 GB for IQ2_XS) to a faster drive, for STRATA_EXPERTS_HEAD
# (gamer-16gb/apply-tuning.py --experts-copy DEST_FILE). Use it when Strata's data folder is on a SATA SSD and you
# have room on an NVMe SSD: on the measured PC, moving the last 42% of experts.bin from the SATA SSD to the NVMe took
# long-context decode from 16.0 to 19.7 tok/s and prefill up 44-71%.
#
# Run it as yourself, not root: the copy must be owned by the user who runs Strata. Linux mincore() reports every page
# of a file you neither own nor can write as cached, and the engine then never schedules its prioritized drive reads.
# On a Windows (NTFS) partition, mount it with ntfs3 and uid=$(id -u),gid=$(id -g) so the file belongs to you.
# The engine compares 16 pages of the copy with experts.bin at every start and falls back to experts.bin if they
# differ, so a stale copy costs speed, never correctness. Re-run this after re-packing the model.
set -eu
[ $# -eq 2 ] || { sed -n 2,3p "$0"; exit 2; }
SRC=$1/experts.bin; DST=$2
[ "$(id -u)" = 0 ] && { echo "run this as the user who runs Strata, not root (see the note at the top)"; exit 1; }
[ -f "$SRC" ] || { echo "$SRC: no such file"; exit 1; }
pgrep -x strata >/dev/null && { echo "stop Strata first"; exit 1; }
total=$(stat -c %s "$SRC")
have=$( [ -f "$DST" ] && stat -c %s "$DST" || echo 0 )
[ "$have" -gt "$total" ] && { echo "$DST is bigger than experts.bin: remove it first"; exit 1; }
mkdir -p "$(dirname "$DST")"
free=$(df -B1 --output=avail "$(dirname "$DST")" | tail -1)
need=$(( total - have ))
echo "experts.bin: $total bytes; the copy has $have; $need to go; $free free on the target"
[ "$free" -gt $(( need + 2000000000 )) ] || { echo "not enough room (keeping 2 GB free)"; exit 1; }
if [ "$need" -gt 0 ]; then   # resumes an interrupted copy
  dd if="$SRC" of="$DST" bs=16M iflag=skip_bytes,count_bytes oflag=seek_bytes,direct conv=notrunc \
     skip="$have" seek="$have" count="$need" status=progress
  sync
fi
echo "verifying (reads both files once) ..."
cmp "$DST" "$SRC" && echo "IDENTICAL. Next: python3 gamer-16gb/apply-tuning.py <setup's strata-*.json> --experts-copy $DST"
