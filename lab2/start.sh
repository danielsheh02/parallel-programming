#!/bin/bash

SRC="lab2.cpp"
EXE_OMP="a.out.omp"
EXE_SEQ="a.out.seq"

g++ -O3 -fopenmp $SRC -o $EXE_OMP
g++ -O3 $SRC -o $EXE_SEQ

schedules=("static" "dynamic" "guided")
# оптимально (n/num_thread) / 100 (1000)
chunks=(12 50 100 200 500 1000 2000)

echo "N,schedule,chunk,time_ms,result_X" > results_omp.csv
echo "N,time_ms,result_X" > results_seq.csv

./$EXE_OMP --file results_omp.csv

for sched in "${schedules[@]}"; do
  for ch in "${chunks[@]}"; do
    ./$EXE_OMP --schedule $sched --chunk $ch --file results_omp.csv
  done
done

./$EXE_SEQ --file results_seq.csv
