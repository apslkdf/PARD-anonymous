"""
    Some handy functions for pytroch model training ...
"""
import torch
import logging
import numpy as np
import scipy.sparse as sp
import pandas as pd
import copy
import random
import os
import datetime
import math


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)

def result2str(metric, Ks, results):
    return '{}@{} = {:.6f}, {}@{} = {:.6f}'.format(
                metric, Ks[0], results[0],
                metric, Ks[1], results[1])\


def norm_tensor(data):
    user_emb = data[:, :64]
    item_emb = data[:, 64:]
    user_emb_normalized = F.normalize(user_emb, dim=1)
    item_emb_normalized = F.normalize(item_emb, dim=1)
    normalized_data = torch.cat([user_emb_normalized, item_emb_normalized], dim=1)
    return normalized_data
# Hyper params
# def use_cuda(config, enabled, device_id=0):
#     if enabled:
#         assert torch.cuda.is_available(), 'CUDA is not available'
#         # torch.cuda.set_device(device_id)
#         config['device'] = 'cuda:{}'.format(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def initLogging():
    """Init for logging
    """
    
    path = 'log/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logFilename = os.path.join(path, current_time+'.txt')

    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin != 0] = 1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin, 0)
    idf = np.log(row / (1 + idf))
    idf = sp.spdiags(idf, 0, col, col)
    return tf * idf


def load_cold_start_data(data_path):
    """load data, including train interaction, validation interaction, test interaction and item raw features"""
    item_content_file = data_path + '/item_features.npy'
    item_content = np.load(item_content_file)

    train_file = data_path + '/train.csv'
    train = pd.read_csv(train_file, dtype=np.int32)
    user_ids = list(set(train['uid'].values))
    train_item_ids = list(set(train['iid'].values))
    train_item_content = item_content[train_item_ids]
    train_item_ids_map = {iid: i for i, iid in enumerate(train_item_ids)}
    for i in train_item_ids_map.keys():
        train['iid'].replace(i, train_item_ids_map[i], inplace=True)

    test_file = data_path + '/test.csv'
    test = pd.read_csv(test_file, dtype=np.int32)
    test_item_ids = list(set(test['iid'].values))
    test_item_content = item_content[test_item_ids]
    test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}

    vali_file = data_path + '/vali.csv'
    vali = pd.read_csv(vali_file, dtype=np.int32)
    vali_item_ids = list(set(vali['iid'].values))
    vali_item_content = item_content[vali_item_ids]
    vali_item_ids_map = {iid: i for i, iid in enumerate(vali_item_ids)}

    data_dict = {'train': train, 'train_item_content': train_item_content, 'user_ids': user_ids,
                 'vali': vali, 'vali_item_content': vali_item_content, 'vali_item_ids_map': vali_item_ids_map,
                 'test': test, 'test_item_content': test_item_content, 'test_item_ids_map': test_item_ids_map,
                 }
    return data_dict


def load_data(config):
    # Load Data
    dataset_dir = "../data/" + config['dataset'] + "/" + "ratings.dat"
    if config['dataset'] == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "ml-100k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "lastfm-2k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "amazon":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    elif config['dataset'] == 'douban':
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    elif config['dataset'] == 'bookcrossing':
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    elif config['dataset'] == "ali-ads":
        rating = pd.read_csv(dataset_dir, sep=",", engine='python')
    else:
        pass
    
    # Reindex
    if config['dataset'] != "ali-ads":
        user_id = rating[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        rating = pd.merge(rating, user_id, on=['uid'], how='left')
        item_id = rating[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        rating = pd.merge(rating, item_id, on=['mid'], how='left')
    else:
        # ali-ads 不需要reindex
        rating['userId'] = rating['uid']
        rating['itemId'] = rating['mid']

    if config['dataset'] == 'douban' or config['dataset'] == 'bookcrossing':
        rating = rating[['userId', 'itemId', 'rating']]
    else:
        rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

    return rating


def negative_sampling(train_data, num_negatives):
    """sample negative instances for training, refer to Heater."""
    # warm items in training set.
    item_warm = np.unique(train_data['iid'].values)
    # arrange the training data with form {user_1: [[user_1], [user_1_item], [user_1_rating]],...}.
    train_dict = {}
    single_user, user_item, user_rating = [], [], []
    grouped_train_data = train_data.groupby('uid')
    for userId, user_train_data in grouped_train_data:
        temp = copy.deepcopy(item_warm)
        for row in user_train_data.itertuples():
            single_user.append(int(row.uid))
            user_item.append(int(row.iid))
            user_rating.append(float(1))
            temp = np.delete(temp, np.where(temp == row.iid))
            for i in range(num_negatives):
                single_user.append(int(row.uid))
                negative_item = np.random.choice(temp)
                user_item.append(int(negative_item))
                user_rating.append(float(0))
                temp = np.delete(temp, np.where(temp == negative_item))
        train_dict[userId] = [single_user, user_item, user_rating]
        single_user = []
        user_item = []
        user_rating = []
    return train_dict


def compute_metrics(evaluate_data, test_scores, negative_scores, Ks):
    test_users, test_items = evaluate_data[0].cpu().data.view(-1).tolist(), evaluate_data[1].cpu().data.view(-1).tolist()
    neg_users, neg_items = evaluate_data[2].cpu().data.view(-1).tolist(), evaluate_data[3].cpu().data.view(-1).tolist()
    tst_scores, neg_scores = test_scores.data.view(-1).tolist(), negative_scores.data.view(-1).tolist()
    # the golden set
    test = pd.DataFrame({'user': test_users,
                        'test_item': test_items,
                        'test_score': tst_scores})
    # the full set
    full = pd.DataFrame({'user': neg_users + test_users,
                        'item': neg_items + test_items,
                        'score': neg_scores + tst_scores})
    full = pd.merge(full, test, on=['user'], how='left')
    # rank the items according to the scores for each user
    full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
    full.sort_values(['user', 'rank'], inplace=True)
    recall, precision, ndcg = [], [], []
    for at_k in Ks:
        top_k = full[full['rank']<=at_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()  # golden items hit in the top_K items
        rec_k = len(test_in_top_k) * 1.0 / full['user'].nunique()
        # pre_k = len(test_in_top_k) * 1.0 / at_k
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        ndcg_k = test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
        recall.append(rec_k)
        ndcg.append(ndcg_k)
        
    return recall, ndcg

def compute_metrics_cold(evaluate_data, user_item_preds, item_ids_map, recall_k):
    """compute evaluation metrics for cold-start items."""
    """input:
    evaluate_data: (uid, iid) dataframe.
    user_item_preds: cold-start item prediction for each user.
    item_ids_map: {ori_id: reindex_id} dict.
    recall_k: top_k metrics.
       output:
    recall, precision, ndcg
    """
    pred = []
    target_rows, target_columns = [], []
    temp = 0
    for uid in user_item_preds.keys():
        # predicted location for each user.
        user_pred = user_item_preds[uid]
        _, user_pred_all = user_pred.topk(k=recall_k[-1])
        user_pred_all = user_pred_all.cpu()
        pred.append(user_pred_all.tolist())

        # cold-start items real location for each user.
        user_cs_items = list(evaluate_data[evaluate_data['uid']==uid]['iid'].unique())
        # record sparse target matrix indexes.
        for item in user_cs_items:
            target_rows.append(temp)
            target_columns.append(item_ids_map[item])
        temp += 1
    pred = np.array(pred)
    target = sp.coo_matrix(
        (np.ones(len(evaluate_data)),
         (target_rows, target_columns)),
        shape=[len(pred), len(item_ids_map)]
    )
    recall, precision, ndcg = [], [], []
    idcg_array = np.arange(recall_k[-1]) + 1
    idcg_array = 1 / np.log2(idcg_array + 1)
    idcg_table = np.zeros(recall_k[-1])
    for i in range(recall_k[-1]):
        idcg_table[i] = np.sum(idcg_array[:(i + 1)])
    for at_k in recall_k:
        preds_k = pred[:, :at_k]
        x = sp.lil_matrix(target.shape)
        x.rows = preds_k
        x.data = np.ones_like(preds_k)

        z = np.multiply(target.todense(), x.todense())
        recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(target, 1))))
        precision.append(np.mean(np.sum(z, 1) / at_k))

        x_coo = sp.coo_matrix(x.todense())
        rows = x_coo.row
        cols = x_coo.col
        target_csr = target.tocsr()
        dcg_array = target_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:at_k].reshape((1, -1)), axis=1)
        idcg = np.sum(target, axis=1) - 1
        idcg[np.where(idcg >= at_k)] = at_k - 1
        idcg = idcg_table[idcg.astype(int)]
        ndcg.append(np.mean(dcg / idcg))

    return recall, precision, ndcg


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def save_log_result(config, test_recalls, test_ndcgs, final_test_round):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    strr = current_time + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
        + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
        + 'best_round: ' + str(final_test_round)
    sstrr = ''
    for k in config.keys():
        sstr = ', ' + f'{k}: ' + str(config[k])
        sstrr += sstr
        # 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr_client: ' + str(config['lr_client']) \
        # + '-' + 'local_epoch: ' + str(config['local_epoch']) + '-' + \
        # '-' + 'client_model_layers: ' + str(config['client_model_layers']) + '-' \
        # 'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
        # + '-' + 'negatives: ' + str(config['num_negative']) + '-' + 'lr_eta: ' + str(config['lr_eta']) + '-' + \
        # 'batch_size: ' + str(config['batch_size']) + '-'  + '-' + 'recall_k: ' + str(config['recall_k']) + '-' + \
        # 'optimizer: ' + config['optimizer'] + '-' + 'l2_regularization: ' + str(config['l2_regularization'])
    strr += sstrr
    file_name = "sh_result/"+config['dataset']+".txt"
    with open(file_name, 'a') as file:
        file.write(strr + '\n')

    # logging.info('fedcs')
    logging.info('recall_list: {}'.format(test_recalls))
    # logging.info('precision_list: {}'.format(test_precisions))
    logging.info('ndcg_list: {}'.format(test_ndcgs))
    # logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr_client: {}, local_epoch: {},'
    #             'client_model_layers: {}, recall_k: {}, dataset: {}, '
    #             'factor: {}, negatives: {}'.format(config['clients_sample_ratio'], config['lr_eta'],
    #                                                         config['batch_size'], config['lr_client'],
    #                                                         config['local_epoch'],
    #                                                         config['client_model_layers'],
    #                                                         config['recall_k'], config['dataset'], config['latent_dim'],
    #                                                         config['num_negative']))
    logging.info('config: {}'.format(sstrr))
    logging.info('Best test recall: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                    # test_precisions[final_test_round],
                                                                                    test_ndcgs[final_test_round],
                                                                                    final_test_round))
    logging.info('\n')


def smooth_labels(labels, num_classes, smoothing=0.1):
    """
    Apply label smoothing to the labels.

    Args:
    labels (torch.Tensor): Original labels, with shape (batch_size,).
    num_classes (int): Number of classes.
    smoothing (float): Smoothing factor.

    Returns:
    torch.Tensor: Smoothed labels with shape (batch_size, num_classes).
    """
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (num_classes - 1)
    
    # One-hot encode the labels
    one_hot = torch.zeros(labels.size(0), num_classes).to(labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), confidence)
    one_hot += smoothing_value
    
    return one_hot

def generate_negative_sample_labels(positive_labels, num_negative_samples, num_classes):
    """
    Generate labels for negative samples.

    Args:
    positive_labels (torch.Tensor): Original positive labels with shape (batch_size,).
    num_negative_samples (int): Number of negative samples to generate for each positive sample.
    num_classes (int): Number of classes including the new class for negative samples.

    Returns:
    torch.Tensor: Negative sample labels with shape (batch_size * num_negative_samples,).
    """
    # Create labels for negative samples, assumed to be a new class index
    neg_class_index = num_classes - 1  # Typically the next class index, e.g., 3 if classes are 0, 1, 2
    neg_labels = torch.full((positive_labels.size(0) * num_negative_samples,), neg_class_index, dtype=torch.long).to(positive_labels.device)
    
    return neg_labels

def ldp_add_noise(model, ep, delta, device):
    for name, param in model.named_parameters():
        if param.grad != None:
            # 裁剪梯度
            noise_scale = 2 * delta / ep
            noise = torch.from_numpy(np.random.laplace(0, noise_scale, param.grad.data.size()).astype(np.float32)).to(device)
            param.grad.data += noise
