#!/usr/bin/env bash
set -x #echo on
#support=(0.1 0.25 0.5 0.9)
support=(0.9)
logs=("data/benchmarking/mining/bpic_2019/logs/10000.xes" )
#logs=("data/benchmarking/mining/bpic_2019/logs/10.xes" "data/benchmarking/mining/bpic_2019/logs/100.xes" "data/benchmarking/mining/bpic_2019/logs/1000.xes" "data/benchmarking/mining/bpic_2019/logs/10000.xes" "data/benchmarking/mining/bpic_2019/logs/100000.xes")
source "/home/giacomo/PycharmProjects/trace_learning/venv/bin/activate"


for l in "${logs[@]}"; do
  for s in "${support[@]}"; do
    python3 /home/giacomo/PycharmProjects/trace_learning/declare_dataless_mining.py -s $s -x "$l" -o "data/benchmarking/mining/results_real_python.csv" -i 2
  done
done

#shutdown -h
