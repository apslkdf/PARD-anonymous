import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import random
import copy
from attacker import *
from config import *
from utils import *
from data import SampleGenerator
# from apdf import *
# from apdf import Client

# config['PRI_TEST_RATIO'] = 0.8

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)


def sample_user(user_list, batch_size=32):
    return random.sample(user_list, batch_size)

def random_attack(config, attr):
    if config["dataset"] == 'douban':
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/user.dat', sep=' ', header=None, names=['uid','location'])
    else:
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/users.dat')

    labels = user_attr[attr]
    if attr == 'gender':
        labels = torch.tensor(labels, dtype=torch.float)
    else:
        labels = torch.tensor(labels)

    user_list = np.array(list(range(config['num_users'])))
    user_train, user_test = train_test_split(user_list, test_size=config['PRI_TEST_RATIO'], random_state=1) # train:test = 2:8
    # user_test, user_valid = train_test_split(user_test, test_size=0.5, random_state=1)
    
    train_labels = labels[user_train]
    max_label = int(max(labels))
    train_label_distribution = np.bincount(train_labels, minlength=max_label) / len(train_labels)
    # print(train_label_distribution)
    # print(labels)
    # print(list(range(max_label + 1)))
    # print(len(list(range(max_label))), train_label_distribution.size)
    test_predictions = np.random.choice(list(range(max_label + 1)), size=len(user_test), p=train_label_distribution)

    if attr == 'gender':
        auc_score = roc_auc_score(labels[user_test], test_predictions)
        msg = '{} discriminator prediction auc score: {}'.format(attr, auc_score)
        print(msg)
    else:
        f1 = f1_score(labels[user_test], test_predictions, average='micro')
        msg = '{} discriminator prediction f1 score: {}'.format(attr, f1)
        print(msg)

    
def privacy_estimator_train(config, attr, discriminator, user_attr, embed, user_train, user_test):
    # device = config['device']
    labels = user_attr[attr]
    if attr == 'gender':
        labels = torch.tensor(labels, dtype=torch.float).to(config['device'])
    else:
        labels = torch.tensor(labels).to(config['device'])
    
    if config['dataset'] == 'ali-ads':
        pri_lr = 0.05
    else:
        pri_lr = 0.05

    if attr == 'gender':
        model = discriminator
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=pri_lr, momentum=0.9)
    elif attr == 'age':
        model = discriminator
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=pri_lr, momentum=0.9)
    elif attr == 'occupation':
        model = discriminator
        if config['dataset'] == 'ali-ads':
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=pri_lr, momentum=0.9)
    elif attr == 'location':
        model = discriminator
        criterion = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    else:
        print('no {} attribute found'.format(attr))
        return
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    model.train()
    user_embed = embed.to(config['device'])
    # user_test, user_valid = train_test_split(user_test, test_size=0.25, random_state=1) # train:val:test = 2:2:6
    best_record = 0
    best_model = None
    best_epoch = 0
    # weight = np.array(weight)
    # user_train_weight = weight[user_train]
    record_loss = []
    for epoch in range(config['pri_epoch']):
        model.train()
        pri_batch = len(user_train.tolist())
        if config['pri_batch'] < pri_batch:
            pri_batch = config['pri_batch']
        users_batch = sample_user(user_train.tolist(), pri_batch)
        # users_batch = user_train[list(WeightedRandomSampler(user_train_weight, 1024))]
        # users_batch = sample_user(list(range(data_generator.n_users)), 3000)
        embed_batch = user_embed[users_batch]
        label_batch = labels[users_batch]
        output = model(embed_batch)
        loss = criterion(output.squeeze(), label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test
        model.eval()
        score, predict = model.predict(user_embed[user_test])
        record_loss.append(criterion(model(user_embed[user_test]).squeeze(), labels[user_test]).detach().cpu())
        if attr == 'gender':
            record = roc_auc_score(labels[user_test].cpu(), score.cpu())
        else:
            record = f1_score(labels[user_test].cpu(), predict.cpu(), average='micro')
            # acc_rec = labels[user_test].cpu() == predict.cpu()
            # print(record)
            
        if record > best_record:
            best_record = record
            best_epoch = epoch

    # best_model.eval()
    # score, predict = best_model.predict(user_embed[user_test])
    if attr == 'gender':
        # auc_score = roc_auc_score(labels[user_test].cpu(), score.cpu())
        msg = '{} discriminator prediction auc score: {:.4f}'.format(attr, best_record)
        # print(msg)
    else:
        # f1 = f1_score(labels[user_test].cpu(), predict.cpu(), average='micro')
        msg = '{} discriminator prediction f1 score: {:.4f}'.format(attr, best_record)
        # print(msg)
    return best_record, best_epoch
    # print(best_epoch)
    # logging.info(msg)
    # logging.info('best epoch: {}'.format(best_epoch))

def get_l(name):
    if name == 'embedding_user.weight':
        l = 0
    elif name == 'embedding_item.weight':
        l = 1
    elif name.split('.')[0] == 'fc_layers':
        l = 2
    else:
        l = 3
    return l

def get_attack_param(config, all_param, all_train_data, attack_modes, client_keys, server_keys):
    # attack_modes: [u_emb, i_emb, mlp]
    U_EMB = 'embedding_puser.weight'
    I_EMB = 'embedding_item.weight'
    user_grads = []
    for user in range(config['num_users']):
        _, items, ratings = torch.LongTensor(all_train_data[0][user]),\
                            torch.LongTensor(all_train_data[1][user]), \
                            torch.FloatTensor(all_train_data[2][user])
        ratings = ratings.float()
        # user_model.train()    

        if config['use_cuda'] is True:
            items, ratings = items.to(config['device']), ratings.to(config['device'])

        client_model_params = all_param['client']
        server_model_param = all_param['server']
        user_param_dict = {}
        if config['pretrain']:
            # load client local param
            for key in client_model_params[user].keys():
                user_param_dict[key] = copy.deepcopy(client_model_params[user][key].data).to(config['device'])
        else:
            # load client local param
            for key in client_keys:
                user_param_dict[key] = copy.deepcopy(client_model_params[user][key].data).to(config['device'])
            # load global param from server
            for key in server_keys:
                user_param_dict[key] = copy.deepcopy(server_model_param[key].data).to(config['device'])
 
        grads = []
        for name in user_param_dict.keys():
            param = user_param_dict[name]
            if name == U_EMB and 'u_emb' in attack_modes:
                grads.append(param.data)
            elif name == I_EMB and 'i_emb' in attack_modes:
                pos_mask = ratings > 1e-5
                # only pos items 
                grads.append(torch.sum(param.data[items][pos_mask], dim=0) / sum(pos_mask))
            elif name.split('.')[0] in ['fc_layers', 'affine_output'] and 'mlp' in attack_modes:
                grads.append(param.data)

        flat_grad = torch.cat([grad.view(-1) for grad in grads if grad is not None])
        user_grads.append(flat_grad)
    user_grads = torch.stack(user_grads, dim=0)
    # print(user_grads)
    return user_grads

def get_all_attack_param(config, all_param, all_train_data, attack_modes, client_keys, server_keys, client_model=None):
    # attack_modes: [u_emb, i_emb, mlp]
    U_EMB = 'embedding_puser.weight'
    I_EMB = 'embedding_item.weight'
    euser_grads, puser_grads, item_grads, eu_i_grads = [], [], [], []
    for user in range(config['num_users']):

        client_model_params = all_param['client']
        server_model_param = all_param['server']

        # load client local param
        if user in client_model_params.keys():
            user_param_dict = {}
            for key in client_keys:
                user_param_dict[key] = copy.deepcopy(client_model_params[user][key].data).to(config['device'])
        else:
            user_param_dict = copy.deepcopy(client_model.state_dict())
        
        # load global param from server
        for key in server_keys:
            if key in server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(server_model_param[key].data).to(config['device'])
 
        eu_grads, pu_grads, ei_grads, i_grads = [], [], [], []
        for name in user_param_dict.keys():
            param = user_param_dict[name]
            if name == 'embedding_euser.weight':
                eu_grads.append(copy.deepcopy(param.data))
                ei_grads.append(copy.deepcopy(param.data))
            elif name == 'embedding_puser.weight':
                pu_grads.append(copy.deepcopy(param.data))
        
        if user in client_model_params.keys():
            ei_grads.append(copy.deepcopy(client_model_params[user][config['ITEM_NAME']]).to(config['device']))
            i_grads.append(copy.deepcopy(client_model_params[user][config['ITEM_NAME']]).to(config['device']))
        else:
            item_emb = user_param_dict['embedding_item.weight'].data[0].to(config['device'])
            ei_grads.append(item_emb)
            i_grads.append(item_emb)
        euflat_grad = torch.cat([grad.view(-1) for grad in eu_grads if grad is not None])
        puflat_grad = torch.cat([grad.view(-1) for grad in pu_grads if grad is not None])
        eiflat_grad = torch.cat([grad.view(-1) for grad in ei_grads if grad is not None])
        iflat_grad = torch.cat([grad.view(-1) for grad in i_grads if grad is not None])
        euser_grads.append(euflat_grad)
        puser_grads.append(puflat_grad)
        eu_i_grads.append(eiflat_grad)
        item_grads.append(iflat_grad)
    euser_grads = torch.stack(euser_grads, dim=0)
    puser_grads = torch.stack(puser_grads, dim=0)
    eu_i_grads = torch.stack(eu_i_grads, dim=0)
    item_grads = torch.stack(item_grads, dim=0)
    # print(user_grads)
    return euser_grads, puser_grads, eu_i_grads, item_grads


def get_all_esti_param(config, all_param):
    # attack_modes: [u_emb, i_emb, mlp]
    age_param, gen_param, occ_param = [], [], []
    for user in range(config['num_users']):
        
        client_model_params = all_param['client']
        server_model_param = all_param['server']

        age_param.append(copy.deepcopy(client_model_params[user][config['ESTI_NAME']+'age']).to(config['device']))
        gen_param.append(copy.deepcopy(client_model_params[user][config['ESTI_NAME']+'gen']).to(config['device']))
        occ_param.append(copy.deepcopy(client_model_params[user][config['ESTI_NAME']+'occ']).to(config['device']))

    age_param = torch.stack(age_param, dim=0)
    gen_param = torch.stack(gen_param, dim=0)
    occ_param = torch.stack(occ_param, dim=0)
    # print(user_grads)
    return age_param, gen_param, occ_param

def esti_dim(config):
    emb_dim = config['latent_dim'] * 2
    pu_emb_dim = config['latent_dim']
    out_dims = [3, 1, 21]
    attack_dims = []
    for out_dim in out_dims:
        layers = [int(emb_dim/2), out_dim]
        pu_layers = [int(pu_emb_dim/2), out_dim]
        # layers = [emb_dim, emb_dim, int(emb_dim/2), out_dim]
        # pu_layers = [pu_emb_dim, pu_emb_dim, int(pu_emb_dim/2), out_dim]
        attack_dim = 0
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            attack_dim += (in_size) * out_size # dim of weight & bias
        for in_size, out_size in zip(pu_layers[:-1], pu_layers[1:]):
            attack_dim += (in_size) * out_size # dim of weight & bias
        attack_dims.append(attack_dim)
    return attack_dims

def eval_privacy(config, all_param, all_train_data, client_keys, server_keys):
    seed_all(config['seed'])
    # eval_param_names = ['u_emb','i_emb', 'mlp']
    if config["dataset"] == 'douban':
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/user.dat', sep=' ', header=None, names=['uid','location'])
    else:
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/users.dat')

    eval_param_names = config['attack_mode'].split('+')
    attack_dim = 0
    if 'u_emb' in eval_param_names:
        attack_dim += config['latent_dim']
    if 'i_emb' in eval_param_names:
        attack_dim += config['latent_dim']
    if 'mlp' in eval_param_names:
        for in_size, out_size in zip(config['client_model_layers'][:-1], config['client_model_layers'][1:]):
            attack_dim += (in_size + 1) * out_size # dim of weight & bias
        attack_dim += config['client_model_layers'][-1] + 1 # dim of the last layer
        # print(attack_dim)
    
    attack_input = get_attack_param(config, all_param, all_train_data, eval_param_names, client_keys, server_keys)
    user_list = np.array(list(range(config['num_users'])))
    user_train, user_test = train_test_split(user_list, test_size=config['PRI_TEST_RATIO'], random_state=1) # train:test = 2:8

    if config['dataset'] == 'ml-100k' or config['dataset'] == 'ml-1m' or config['dataset'] == 'ali-ads':
        age_discriminator = Discriminator(config, attack_dim, attr='age').cuda()
        occupation_discriminator = Discriminator(config, attack_dim, attr='occupation').cuda()
        gender_discriminator = Discriminator(config, attack_dim, attr='gender').cuda()

        age_score, age_epoch = privacy_estimator_train(config, 'age', age_discriminator, user_attr, attack_input, user_train, user_test)
        gen_score, gen_epoch = privacy_estimator_train(config, 'gender', gender_discriminator, user_attr, attack_input, user_train, user_test)
        occ_score, occ_epoch = privacy_estimator_train(config, 'occupation', occupation_discriminator, user_attr, attack_input, user_train, user_test)
        print('attack mode: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}], epooch: [{},{},{}]'.format(config['attack_mode'], age_score, gen_score, occ_score, age_epoch, gen_epoch, occ_epoch))
        logging.info('attack mode: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}], epooch: [{},{},{}]'.format(config['attack_mode'], age_score, gen_score, occ_score, age_epoch, gen_epoch, occ_epoch))
    
    elif config['dataset'] == 'douban':
        loc_discriminator = Discriminator(config, attack_dim, attr='location').cuda()
        loc_score, loc_epoch = privacy_estimator_train(config, 'location', loc_discriminator, user_attr, attack_input, user_train, user_test)
        print('attack mode: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format(config['attack_mode'], loc_score,loc_epoch))
        logging.info('attack mode: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format(config['attack_mode'], loc_score,loc_epoch))


def privacy_eval(config, model, user_embed, user_test, attr, user_attr):
    model.eval()
    labels = user_attr[attr]
    if attr == 'gender':
        labels = torch.tensor(labels, dtype=torch.float).to(config['device'])
    else:
        labels = torch.tensor(labels).to(config['device'])
    score, predict = model.predict(user_embed[user_test])
    if attr == 'gender':
        record = roc_auc_score(labels[user_test].cpu(), score.cpu())
    else:
        record = f1_score(labels[user_test].cpu(), predict.cpu(), average='micro')
    return record

def eval_all_privacy(config, all_param, all_train_data, client_keys, server_keys, client_model=None):
    seed_all(config['seed'])
    # eval_param_names = ['u_emb','i_emb', 'mlp']
    if config["dataset"] == 'douban':
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/user.dat', sep=' ', header=None, names=['uid','location'])
    else:
        user_attr = pd.read_csv(f'../data/{config["dataset"]}/users.dat')

    eval_param_names = config['attack_mode'].split('+')
    eu_attack_dim = config['latent_dim']
    pu_attack_dim = config['latent_dim']
    ei_attack_dim = config['latent_dim'] * 2

    user_grads, puser_grads, eu_i_grads, item_grads = get_all_attack_param(config, all_param, None, eval_param_names, client_keys, server_keys, client_model)
    attack_inputs = [user_grads, puser_grads, eu_i_grads]
    attack_dims = [eu_attack_dim, pu_attack_dim, ei_attack_dim]
    attack_mode = ['euser', 'puser', 'euser_item']

    user_list = np.array(list(range(config['num_users'])))
    user_train, user_test = train_test_split(user_list, test_size=config['PRI_TEST_RATIO'], random_state=1) # train:test = 2:8

    # eval rec model param
    for i, (attack_dim, attack_input) in enumerate(zip(attack_dims, attack_inputs)):
        seed_all(config['seed'])
        if config['dataset'] == 'ml-100k' or config['dataset'] == 'ml-1m':
            attrs = ['age', 'gender', 'occupation']
            esti_attrs = attrs
        elif config['dataset'] == 'ali-ads':        
            attrs = ['age', 'gender']
            esti_attrs = attrs
        elif config['dataset'] == 'douban':
            attrs = ['location']
            esti_attrs = attrs
        elif config['dataset'] == 'bookcrossing':  
            attrs = ['age', 'location'] 
            esti_attrs = ['bk_age', 'bk_location']

        pri_scores = []
        pri_epochs = []
        for esti_attr, attr in zip(esti_attrs, attrs):
            discriminator = Discriminator(config, attack_dim, attr=esti_attr).to(config['device'])
            score, epoch = privacy_estimator_train(config, attr, discriminator, user_attr, attack_input, user_train, user_test)
            pri_scores.append('{:.4f}'.format(score))
            pri_epochs.append(epoch)
        print('attack mode: [{}], attack score[F1,F1]: {}, epooch: {}'.format(attack_mode[i], pri_scores, pri_epochs))
        logging.info('attack mode: [{}], attack score[F1,F1]: {}, epooch: {}'.format(attack_mode[i], pri_scores, pri_epochs))


    # # attack estimator param
    # if config['dataset'] == 'ml-100k':
    #     age_param, gen_param, occ_param = get_all_esti_param(config, all_param)
    #     age_dim, gen_dim, occ_dim = esti_dim(config)

    #     age_discriminator = Discriminator(config, age_dim, attr='age').to(config['device'])
    #     occupation_discriminator = Discriminator(config, occ_dim, attr='occupation').to(config['device'])
    #     gender_discriminator = Discriminator(config, gen_dim, attr='gender').to(config['device'])
    #     age_score = privacy_estimator_train(config, 'age', age_discriminator, user_attr, age_param, user_train, user_test)
    #     gen_score = privacy_estimator_train(config, 'gender', gender_discriminator, user_attr, gen_param, user_train, user_test)
    #     occ_score = privacy_estimator_train(config, 'occupation', occupation_discriminator, user_attr, occ_param, user_train, user_test)
    #     print('attack mode: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('esti param', age_score, gen_score, occ_score))
    #     logging.info('attack mode: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('esti param', age_score, gen_score, occ_score))

            
def load_estimator(config, server_param):
    if config['user_only']:
        age_param, gen_param, occ_param = {}, {}, {}
        for key in server_param:
            if key.split('.')[0] == 'age_estimator':
                esti_key = key.split('.', 1)[1]
                age_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'gen_estimator':
                esti_key = key.split('.', 1)[1]
                gen_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'occ_estimator':
                esti_key = key.split('.', 1)[1]
                occ_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
        return age_param, gen_param, occ_param
    else:
        age_param, gen_param, occ_param = {}, {}, {}
        pu_age_param, pu_gen_param, pu_occ_param = {}, {}, {}
        for key in server_param:
            if key.split('.')[0] == 'age_estimator':
                esti_key = key.split('.', 1)[1]
                age_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'gen_estimator':
                esti_key = key.split('.', 1)[1]
                gen_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'occ_estimator':
                esti_key = key.split('.', 1)[1]
                occ_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'pu_age_estimator':
                esti_key = key.split('.', 1)[1]
                pu_age_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'pu_gen_estimator':
                esti_key = key.split('.', 1)[1]
                pu_gen_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
            elif key.split('.')[0] == 'pu_occ_estimator':
                esti_key = key.split('.', 1)[1]
                pu_occ_param[esti_key] = copy.deepcopy(server_param[key].data).to(config['device'])
        return age_param, gen_param, occ_param, pu_age_param, pu_gen_param, pu_occ_param

if __name__ == '__main__':
    config = get_config()
    print(config)
    seed_all(config['seed'])
    random_attack(config, 'age')
    random_attack(config, 'gender')
    random_attack(config, 'occupation')
    # use_cuda(True, config['device_id'])
    # # NAME = 'FedNCF'
    # is_ldp = False
    # NAME = 'pretrain-apdf'

    # # user_attr = pd.read_csv(f'../data/{config["dataset"]}/users.dat')
    # path = f'./saved_model/{config["dataset"]}/{NAME}.pkl'
    # all_param = torch.load(path)
    # rating = load_data(config)
    # sample_generator = SampleGenerator(config=config, ratings=rating)
    # # all_train_data = sample_generator.store_all_train_data(config['num_negative'])
    # client_model = Client(config)
    # pptrainer = FedTrainer(config)

    # eval_all_privacy(config, all_param, None, pptrainer.client_keys, pptrainer.server_keys)


    # server_param = all_param['server']
    # client_param = all_param['client']
    # # use estimator attack
    # if config['dataset'] == 'ml-100k':
    #     if 'age_estimator.net.0.weight' in server_param.keys():
    #         age_param, gen_param, occ_param, pu_age_param, pu_gen_param, pu_occ_param = load_estimator(config, server_param)
    #         age_discriminator = Discriminator(config, ei_attack_dim, attr='age').to(config['device'])
    #         occupation_discriminator = Discriminator(config, ei_attack_dim, attr='occupation').to(config['device'])
    #         gender_discriminator = Discriminator(config, ei_attack_dim, attr='gender').to(config['device'])
    #         pu_age_discriminator = Discriminator(config, pu_attack_dim, attr='age').to(config['device'])
    #         pu_occupation_discriminator = Discriminator(config, pu_attack_dim, attr='occupation').to(config['device'])
    #         pu_gender_discriminator = Discriminator(config, pu_attack_dim, attr='gender').to(config['device'])

    #         age_discriminator.load_state_dict(age_param)
    #         gender_discriminator.load_state_dict(gen_param)
    #         occupation_discriminator.load_state_dict(occ_param)
    #         pu_age_discriminator.load_state_dict(pu_age_param)
    #         pu_gender_discriminator.load_state_dict(pu_gen_param)
    #         pu_occupation_discriminator.load_state_dict(pu_occ_param)

    #         age_score = privacy_eval(config, age_discriminator, eu_i_grads, user_test, 'age', user_attr)
    #         gen_score = privacy_eval(config, gender_discriminator, eu_i_grads, user_test, 'gender', user_attr)
    #         occ_score = privacy_eval(config, occupation_discriminator, eu_i_grads, user_test, 'occupation', user_attr)
    #         print('[estimator] attack: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('euser_item', age_score, gen_score, occ_score))
    #         logging.info('[estimator] attack: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('euser_item', age_score, gen_score, occ_score))

    #         age_score = privacy_eval(config, pu_age_discriminator, puser_grads, user_test, 'age', user_attr)
    #         gen_score = privacy_eval(config, pu_gender_discriminator, puser_grads, user_test, 'gender', user_attr)
    #         occ_score = privacy_eval(config, pu_occupation_discriminator, puser_grads, user_test, 'occupation', user_attr)
    #         print('[estimator] attack: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('puser', age_score, gen_score, occ_score))
    #         logging.info('[estimator] attack: [{}], attack score[F1,AUC,F1]: [{:.4f},{:.4f},{:.4f}]'.format('puser', age_score, gen_score, occ_score))
    # elif config['dataset'] == 'douban':
    #     if 'loc_estimator.net.0.weight' in server_param.keys():
    #         loc_param, pu_loc_param = load_estimator(config, server_param)
    #         loc_discriminator = Discriminator(config, ei_attack_dim, attr='location').to(config['device'])
    #         pu_loc_discriminator = Discriminator(config, pu_attack_dim, attr='location').to(config['device'])
    #         loc_discriminator.load_state_dict(loc_param)
    #         pu_loc_discriminator.load_state_dict(pu_loc_param)
            
    #         loc_score, loc_epoch = privacy_estimator_train(config, 'location', loc_discriminator, user_attr, attack_input, user_train, user_test)
    #         print('[estimator] attack: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format('euser_item', loc_score,loc_epoch))
    #         logging.info('[estimator] attack: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format('euser_item', loc_score,loc_epoch))
    #         loc_score, loc_epoch = privacy_estimator_train(config, 'location', pu_loc_discriminator, user_attr, attack_input, user_train, user_test)
    #         print('[estimator] attack: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format('puser', loc_score,loc_epoch))
    #         logging.info('[estimator] attack: [{}], attack score[F1]: [{:.4f}], epooch: [{}]'.format('puser', loc_score,loc_epoch))
            