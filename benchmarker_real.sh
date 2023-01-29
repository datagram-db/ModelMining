support=(0.1 0.25 0.5 0.9)
logs=("/home/sam/Documents/Repositories/CodeBases/knobab/data/benchmarking/mining/bpic_2019/logs/10.xes" "/home/sam/Documents/Repositories/CodeBases/knobab/data/benchmarking/mining/bpic_2019/logs/100.xes"
      "/home/sam/Documents/Repositories/CodeBases/knobab/data/benchmarking/mining/bpic_2019/logs/1000.xes" "/home/sam/Documents/Repositories/CodeBases/knobab/data/benchmarking/mining/bpic_2019/logs/10000.xes"
      "/home/sam/Documents/Repositories/CodeBases/knobab/data/benchmarking/mining/bpic_2019/logs/100000.xes")

source "/home/sam/Documents/Repositories/CodeBases/ModelMining/envs/bin/activate"

for s in "${support[@]}"; do
  for l in "${logs[@]}"; do
    python3 declare_dataless_mining.py -s $s -x "$l" -o "/home/sam/Documents/Repositories/CodeBases/ModelMining/results_real.csv" -i 5
  done
done
