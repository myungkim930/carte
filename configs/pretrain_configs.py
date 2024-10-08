# Basic exp settings
num_steps = 10000
num_batch = 2
num_sample = 128
num_pos = 7
max_rel = 30
num_rel = 4
data_dir = "/data/parietal/store3/work/mkim/codes/carte_dev/data/data_yago/yago3_2022"

# Exp configs (for saving)
exp_configs = dict()
exp_configs["data_name"] = data_dir.split("/")[-1]
exp_configs["num_steps"] = num_steps
exp_configs["num_hops"] = 1
exp_configs["num_sample"] = num_sample
exp_configs["num_batch"] = num_batch
exp_configs["num_pos"] = num_pos
exp_configs["max_rel"] = max_rel
exp_configs["num_rel"] = num_rel

# Preprocessor configs
preprocessor_configs = dict()
constructor = dict()
constructor["data_kg_dir"] = data_dir
constructor["num_hops"] = 1
constructor["num_sample"] = num_sample
constructor["num_pos"] = num_pos
constructor["max_rels"] = max_rel
idx_iterator = dict()
idx_iterator["data_kg_dir"] = data_dir
idx_iterator["num_rel"] = num_rel
idx_iterator["num_batch"] = num_batch
preprocessor_configs["constructor"] = constructor
preprocessor_configs["idx_iterator"] = idx_iterator

# Neural network configs
model_configs = dict()
model_configs["input_dim"] = 300
model_configs["num_heads"] = 10  # Need to have 300 % 10 = 0
model_configs["dim_feedforward"] = 2048
model_configs["num_layers"] = 2
model_configs["dropout"] = 0.1
model_configs["num_batch"] = num_batch
model_configs["num_sample"] = num_sample

# Pretrainer configs
pretrain_configs = dict()
pretrain_configs["model_configs"] = model_configs
pretrain_configs["learning_rate"] = 1e-6
pretrain_configs["warmup_steps"] = 100
pretrain_configs["device"] = "cuda:0"
pretrain_configs["save_every"] = 1000