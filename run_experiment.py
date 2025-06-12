import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from trainer import *
# from fedncf import *
from utils import *
from config import *
from data import SampleGenerator

config = get_config()
# print(config)

seed_all(config['seed'])

if config['pretrain']:
    config['lam_eu'] = 0
    config['lam_pu'] = 0

# Logging, logFilename: ./log/[current_time].txt
initLogging()
logging.info(config)

trainer = FedTrainer(config)

# Load Data, rating[['userId', 'itemId', 'rating', 'timestamp']]
rating = load_data(config)


# DataLoader for training
sample_generator = SampleGenerator(config=config, ratings=rating)

test_recalls, test_ndcgs, final_test_round = trainer.run_experiment(config, sample_generator)

save_log_result(config, test_recalls, test_ndcgs, final_test_round)