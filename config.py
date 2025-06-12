import argparse
import torch

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_config():
    # Training settings
    parser = argparse.ArgumentParser()
    # parser.add_argument('--alias', type=str, default='fedcs')
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=1)
    # parser.add_argument('--server_epoch', type=int, default=1)
    parser.add_argument('--lr_eta', type=int, default=80)
    # parser.add_argument('--reg', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr_client', type=float, default=0.5)
    # parser.add_argument('--lr_server', type=float, default=0.005)
    parser.add_argument('--dataset', type=str, default='ali-ads')
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--num_items', type=int)
    # parser.add_argument('--num_items_vali', type=int)
    # parser.add_argument('--num_items_test', type=int)
    # parser.add_argument('--content_dim', type=int)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_negative', type=int, default=5) # number of negative samples for training
    # parser.add_argument('--server_model_layers', type=str, default='300')
    parser.add_argument('--client_model_layers', type=str, default='128,32') # 2*emb_size, emb_size, 1
    parser.add_argument('--recall_k', type=str, default='10,20')
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--use_cuda', type=boolean_string, default=True)
    parser.add_argument('--device_id', type=str, default='0')
    # parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--NUM_NEG', type=int, default=999) # number of negative samples for val&tst
    parser.add_argument('--earlystop', type=int, default=20) # number of negative samples for val&tst
    # parser.add_argument('--use_diff_lr', type=boolean_string, default=True) # 是否采用user item 与mlp不同的lr，这样训练更快

    # privacy-preserving training config
    parser.add_argument('--train_ppmodel', type=boolean_string, default=False) # privacy-preserving trainer
    parser.add_argument('--save_model', type=boolean_string, default=False)
    parser.add_argument('--save_name', type=str, default='FedNCF.pkl')

    # privacy evaluate
    parser.add_argument('--pri_epoch', type=int, default=150)
    parser.add_argument('--pri_batch', type=int, default=2000) # number of negative samples for val&tst
    parser.add_argument('--attack_mode', type=str, default='u_emb+i_emb', help='["u_emb", "mlp", "i_emb", "u_emb+mlp"]')
    parser.add_argument('--grad_based', type=boolean_string, default=False) # 是否使用grad-based mlp-attacker, 最好False
    
    # privacy-preserving disentaglement methods
    parser.add_argument('--pries_lr', type=float, default=0.1, help='lr of privacy estimator in client')
    parser.add_argument('--pries_epoch', type=int, default=1, help='lr of privacy estimator in client')
    parser.add_argument('--reg', type=float, default=1, help='similarity loss between global and local exclusive user emb')
    parser.add_argument('--lam_eu', type=float, default=5e-1, help='lambda of min I(eu;P)')
    parser.add_argument('--lam_pu', type=float, default=3e-1, help='lambda of max I(pu,P) and min H(pu|P)')
    parser.add_argument('--pretrain', type=boolean_string, default=True, help='pre-train the fedncf, keep all param in client')
    parser.add_argument('--finetune', type=boolean_string, default=False, help='fine tune the fedncf from data')
    parser.add_argument('--NAME', type=str, default='APDF-pretrain-ml.pkl', help='fine tune param')
    parser.add_argument('--pri_ratio', type=str, default='1,1,1', help='importance of age, gender, occupation')
    parser.add_argument('--user_only', type=boolean_string, default=False, help='only protect privacy of exclusive user, this will impact the estimator input')
    parser.add_argument('--ITEM_NAME', type=str, default='updated_item', help='positive item emb updated to server')
    parser.add_argument('--EUSER_NAME', type=str, default='test_euser_emb', help='client euser embed')
    parser.add_argument('--ESTI_NAME', type=str, default='esti_', help='positive item emb updated to server')
    parser.add_argument('--PRI_TEST_RATIO', type=float, default=0.8, help='ratio of test users in privacy test') # 
    parser.add_argument('--GNN', type=boolean_string, default=False, help='use LightGCN(True), NCF(False)')
    parser.add_argument('--gnn_drop', type=float, default=0.5)
    parser.add_argument('--is_esti_local', type=boolean_string, default=True, help='train estimator on trust party (False) or locally (True)')
    parser.add_argument('--pubdata_ratio', type=float, default=0.2, help='estimator train data ratio of trust party')
    parser.add_argument('--localdata_ratio', type=float, default=0.6, help='estimator train data ratio of local data')
    parser.add_argument('--pri_esti_round', type=int, default=1, help='train estimator every x round')


    args = parser.parse_args()


    # Model.
    config = vars(args)
    if len(config['recall_k']) > 1:
        config['recall_k'] = [int(item) for item in config['recall_k'].split(',')]
    else:
        config['recall_k'] = [int(config['recall_k'])]
    # if len(config['server_model_layers']) > 1:
    #     config['server_model_layers'] = [int(item) for item in config['server_model_layers'].split(',')]
    # else:
    #     config['server_model_layers'] = int(config['server_model_layers'])
    if len(config['client_model_layers']) > 1:
        config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')]
    else:
        config['client_model_layers'] = int(config['client_model_layers'])
    
    if len(config['pri_ratio'].split(',')) != 3:
        config['pri_ratio'] = [1,1,1]
    else:
        config['pri_ratio'] = [float(ratio) for ratio in config['pri_ratio'].split(',')]


    if config['dataset'] == 'ml-1m':
        config['num_users'] = 6040
        config['num_items'] = 3706
        config['num_age'] = 3
        config['num_gender'] = 2
        config['num_occupation'] = 21
        config['NUM_NEG'] = 999
        config['gnn_drop'] = 0.1
        config['pri_esti_round'] = 2 # 每隔多少epoch训练一次estimator
    elif config['dataset'] == 'ml-100k':
        config['num_users'] = 943
        config['num_items'] = 1682
        config['num_age'] = 3
        config['num_gender'] = 2
        config['num_occupation'] = 21
        config['NUM_NEG'] = 99
        config['pri_esti_round'] = 1
    # elif config['dataset'] == 'lastfm-2k':
    #     config['num_users'] = 1600
    #     config['num_items'] = 12454
    # elif config['dataset'] == 'amazon':
    #     config['num_users'] = 8072
    #     config['num_items'] = 11830
    # elif config['dataset'] == 'douban':
    #     config['num_users'] = 6368
    #     config['num_items'] = 22347
    # elif config['dataset'] == 'bookcrossing':
    #     config['num_users'] = 4370
    #     config['num_items'] = 9358
    #     config['num_age'] = 4
    #     config['num_location'] = 75
    elif config['dataset'] == 'ali-ads':
        config['num_users'] = 3198
        config['num_items'] = 4282
        config['num_age'] = 7
        config['num_gender'] = 2
        config['num_occupation'] = 2
        config['NUM_NEG'] = 999
        # config['NAME'] = 'apdf-pre2-titan.pkl' # ml
        # config['NAME'] = 'APDF-pretrain-ali.pkl' # ali
        config['gnn_drop'] = 0.1
        # config['pri_esti_round'] = 2 # 每隔多少epoch训练一次estimator
    else:
        pass

    if config['device_id'] != 'cpu' and torch.cuda.is_available():
        config['device'] = 'cuda:{}'.format(config['device_id'])
    else:
        config['device'] = 'cpu'

    return config

if __name__ == '__main__':
    config = get_config()
    print(f'./saved_model/{config["save_name"]}')
