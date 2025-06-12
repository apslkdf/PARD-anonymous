import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from apdf import FedTrainer
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

if config['finetune']:
    path = f'./saved_model/{config["dataset"]}/{config["NAME"]}'
    print(path)
    all_param = torch.load(path)
    trainer.client_model_params = copy.deepcopy(all_param['client'])
    trainer.server_model_param = copy.deepcopy(all_param['server'])
    logging.info('-' * 80)
    logging.info('Testing load param!')
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data
    test_recall, test_ndcg, loss = trainer.fed_evaluate(test_data)

    logging.info(result2str('Recall', config['recall_k'], test_recall))
    logging.info(result2str('NDCG', config['recall_k'], test_ndcg))
    