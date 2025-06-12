import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from apdf import *
from utils import *
from config import *
from data import SampleGenerator
from itertools import product
from trainer import *

config = get_config()

# ml-1m: fedncf:0.5; fedgnn:0.1;
lr_clients = [0.1] if config['GNN'] else [0.1] # lr 5, 10 就不行了 [1, 0.5, 0.05]

local_epochs = [1] # [1, 2]

pri_epochs = [1, 3] if config['dataset'] == 'ml-1m' else [3]
pri_lrs = [0.1] if config['dataset'] == 'ml-100k' else [0.01] # ali-ads&ml1m=0.01; ml100k=0.1;
lr_etas = [1.] if config['dataset'] == 'ml-100k' else [80.]

latent_dims = [64]
neg_samples = [5]

localdata_ratio = [0.6] # local data ratio
pubdata_ratio = [0.08]

# lam_eus = [0.1] # disentangle eu
lam_eus = [0.5] # disentangle eu
# lam_eus = [0.5] # disentangle [0.1, 0.5, 0.7, 0.9]
# lam_pus = [0.5] # disentangle pu
lam_pus = [0, 0.1, 0.3, 0.5, 0.7, 0.9]  # disentangle pu [0.1, 0.3, 0.5] 
clients_sample_ratios = [0.8] if config['dataset'] == 'ml-1m' else [1.]

hypr_param = [lr_clients, local_epochs, lr_etas, latent_dims, neg_samples, lam_eus, lam_pus, pri_lrs, pri_epochs, clients_sample_ratios, localdata_ratio, pubdata_ratio]
hypr_name = ['lr_client', 'local_epoch', 'lr_eta', 'latent_dim', 'num_negative', 'lam_eu', 'lam_pu', 'pries_lr', 'pries_epoch', 'clients_sample_ratio', 'localdata_ratio', 'pubdata_ratio']
param_settings = list(product(*hypr_param))

# Logging, logFilename: ./log/[current_time].txt
initLogging()

for param in param_settings:
    for k, v in zip(hypr_name, param):
        config[k] = v

    logging.info(config)
    seed_all(config['seed'])

    # if config['train_ppmodel']:
    #     trainer = PPFedNCFTrainer(config)
    # else:
    trainer = FedTrainer(config)

    # Load Data, rating[['userId', 'itemId', 'rating', 'timestamp']]
    rating = load_data(config)

    # DataLoader for training
    sample_generator = SampleGenerator(config=config, ratings=rating)
    # all_train_data, val_data, test_data = sample_generator.get_data()

    test_recalls, test_ndcgs, final_test_round = trainer.run_experiment(config, sample_generator)

    save_log_result(config, test_recalls, test_ndcgs, final_test_round)
