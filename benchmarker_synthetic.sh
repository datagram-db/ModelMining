#!/usr/bin/env bash
set -x #echo on
logs=("data/benchmarking/mining/synthetic/logs/10_10_10.xes" "data/benchmarking/mining/synthetic/logs/10_15_10.xes"
      "data/benchmarking/mining/synthetic/logs/10_20_10.xes" "data/benchmarking/mining/synthetic/logs/10_25_10.xes"
      "data/benchmarking/mining/synthetic/logs/10_30_10.xes" "data/benchmarking/mining/synthetic/logs/100_10_10.xes"
      "data/benchmarking/mining/synthetic/logs/100_15_10.xes" "data/benchmarking/mining/synthetic/logs/100_20_10.xes"
      "data/benchmarking/mining/synthetic/logs/100_25_10.xes" "data/benchmarking/mining/synthetic/logs/100_30_10.xes"
      "data/benchmarking/mining/synthetic/logs/1000_10_10.xes" "data/benchmarking/mining/synthetic/logs/1000_15_10.xes"
      "data/benchmarking/mining/synthetic/logs/1000_20_10.xes" "data/benchmarking/mining/synthetic/logs/1000_25_10.xes"
      "data/benchmarking/mining/synthetic/logs/1000_30_10.xes")

source "/home/giacomo/PycharmProjects/trace_learning/venv/bin/activate"

for l in "${logs[@]}"; do
  python3 /home/giacomo/PycharmProjects/trace_learning/declare_dataless_mining.py -s 0.9 -x "$l" -o "data/benchmarking/mining/results_synthetic_python.csv" -i 5
done