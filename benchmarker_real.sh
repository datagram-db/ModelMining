support=(0.1)
logs=("/home/sam/Documents/Repositories/Codebases/knobab/data/benchmarking/mining/bpic_2019/logs/100000.xes")

source "/home/sam/Documents/Repositories/Codebases/ModelMining/venv/bin/activate"

for s in "${support[@]}"; do
  for l in "${logs[@]}"; do
    python3 declare_dataless_mining.py -s $s -x "$l" -o "/home/sam/Documents/Repositories/Codebases/ModelMining/results_real.csv" -i 1
  done
done

shutdown -h