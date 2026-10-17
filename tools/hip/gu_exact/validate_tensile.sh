#!/bin/bash
# validate_tensile.sh <group-dir> <solution-substring> <T...>: bitwise vs rocBLAS's fallback for each T (TrigSin data)
G=$1; SUB=$2; shift 2
SRC=out-sweep1/1_BenchmarkProblems/$G/00_Final/source
IDX=$(python3 -c "
import csv,sys; h=[x.strip() for x in next(csv.reader(open('out-sweep1/2_BenchmarkData/$G.csv')))][13:]
m=[i for i,x in enumerate(h) if x.endswith('$SUB')]; print(m[0] if len(m)==1 else 'AMBIG'+str(m))")
C=${TENSILE_CLIENT:?set TENSILE_CLIENT to the built tensile-client}
W=$(mktemp -d ${TMPDIR:-/tmp}/val.XXXX); cd $W
for T in "$@"; do
  sed -e "s/^problem-size=.*/problem-size=1280,$T,1,2560/; s/^init-a=.*/init-a=TrigSin/; s/^init-b=.*/init-b=TrigSin/; s/^init-alpha=.*/init-alpha=One/; s/^init-beta=.*/init-beta=Zero/; s/^num-enqueues-per-sync=.*/num-enqueues-per-sync=1/; s/^num-syncs-per-benchmark=.*/num-syncs-per-benchmark=1/; s/^num-warmups=.*/num-warmups=0/; s#^results-file=.*#results-file=$W/r.csv#" $OLDPWD/$SRC/ClientParameters.ini > v.ini
  printf "print-tensor-a=True\nprint-tensor-b=True\nprint-tensor-d=True\ndump-tensors=True\n" >> v.ini
  $C --config-file v.ini --solution-start-idx $IDX --num-solutions 1 > c.log 2>&1 || { echo "T=$T client failed"; continue; }
  echo "T=$T $(./tensile_dump_check . 1280 $T 2560 | head -1)"
done
cd /; rm -rf $W
