#!/bin/bash

SRC="lab3.cpp"
EXE_OMP="a.out.omp"
EXE_SEQ="a.out.seq"

g++ -O3 -fopenmp $SRC -o $EXE_OMP
g++ -O3 $SRC -o $EXE_SEQ

# schedules=("static" "dynamic" "guided")
# chunks=(12 50 100 250 500 1000 2000)

# echo "N,schedule,chunk,thread_num,time_ms,result_X" > results_omp.csv
# echo "N,time_ms,result_X" > results_seq.csv

# ./$EXE_OMP --file results_omp.csv

# for sched in "${schedules[@]}"; do
#   for ch in "${chunks[@]}"; do
#     ./$EXE_OMP --schedule $sched --chunk $ch --file results_omp.csv
#   done
# done

# ./$EXE_SEQ --file results_seq.csv

echo "N,schedule,chunk,thread_num,time_ms,result_X" > results_omp_thread.csv

thread_num=(2 3 4 5 6 7 8 9 10 11 12)
for num in "${thread_num[@]}"; do
  ./$EXE_OMP  --file results_omp_thread.csv --thread_num $num
done