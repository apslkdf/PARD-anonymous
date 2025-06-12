import torch
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from eval_privacy import eval_all_privacy
from sklearn.model_selection import train_test_split
from apdf import Client
import time


class Trainer(object):
    """Meta Trainer for training & evaluating NCF model

    Note: Subclass should implement self.client_model and self.server_model!
    """

    def __init__(self, config):
        self.config = config  # model configuration
        # self.server_opt = torch.optim.Adam(self.server_model.parameters(), lr=config['lr_server'],
        #                                    weight_decay=config['l2_regularization'])
        self.device = config['device']
        self.server_model_param = {}
        self.client_model_params = {} # client 保存在local的参数
        self.client_crit = torch.nn.BCELoss()
        if self.config['dataset'] == 'ml-100k' or self.config['dataset'] == 'ali-ads' or self.config['dataset'] == 'ml-1m':
            self.privacy_keys = ['age', 'gender', 'occupation']
            self.privacy_ratios = config['pri_ratio'] # meta-param for adaptive learning

        user_list = np.array(list(range(config['num_users'])))
        pri_user, _ = train_test_split(user_list, test_size=config['PRI_TEST_RATIO'], random_state=1) # users who dont care their privacy
        self.pri_user = pri_user

        # public privacy data
        self.public_uemb = []
        self.public_uiemb = []
        self.public_y = {key:[] for key in self.privacy_keys}
        self.trust_party = Client(config)
        self.trust_party.to(self.device)

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def get_estimator_criterion(self, model_client, attr, is_pu):
        if attr == 'gender':
            if is_pu:
                estimator = model_client.pu_gen_estimator
            else:
                estimator = model_client.gen_estimator
            criterion = torch.nn.BCELoss()
        elif attr == 'age':
            if is_pu:
                estimator = model_client.pu_age_estimator
            else:
                estimator = model_client.age_estimator
            criterion = torch.nn.NLLLoss()
        elif attr == 'occupation':
            if is_pu:
                estimator = model_client.pu_occ_estimator
            else:
                estimator = model_client.occ_estimator
            if self.config['dataset'] == 'ali-ads':
                criterion = torch.nn.BCELoss()
            else:
                criterion = torch.nn.NLLLoss()
        elif attr == 'location':
            if is_pu:
                estimator = model_client.pu_loc_estimator
            else:
                estimator = model_client.loc_estimator
            criterion = torch.nn.NLLLoss()
        return estimator, criterion

    def label2torch(self, attr, esti_input, label):
        if attr == 'gender':
            labels = torch.full((esti_input.shape[0],), float(label)).to(self.device)
        elif self.config['dataset'] == 'ali-ads' and attr == 'occupation':
            labels = torch.full((esti_input.shape[0],), float(label)).to(self.device)
            # labels = torch.tensor([labels]*esti_input.shape[0], dtype=torch.float).to(self.device)
        else:
            labels = torch.full((esti_input.shape[0],), label, dtype=torch.long).to(self.device)
            # labels = torch.tensor([labels]*esti_input.shape[0]).to(self.device)
        return labels

    def get_full_input_label(self, esti_input, labels, attr, is_pu=False, is_full=True):
        if is_pu:
            full_input = copy.deepcopy(self.public_uemb)
        else:
            full_input = copy.deepcopy(self.public_uiemb)
        
        if len(full_input) > 0 and is_full:
            full_input = torch.cat(full_input, dim=0)
            idx = np.random.choice(full_input.shape[0], int(full_input.shape[0]*self.config['pubdata_ratio']), replace=False)
            if idx.shape[0] > 0:
                sub_input = full_input[idx]
            else:
                sub_input = full_input
            full_input = torch.cat([sub_input, esti_input.cpu()], dim=0)
        
            full_label = copy.deepcopy(self.public_y[attr])
            full_label = torch.cat(full_label, dim=0)
            if idx.shape[0] > 0:
                sub_label = full_label[idx]
            else:
                sub_label = full_label
            full_label = torch.cat([sub_label, labels], dim=0)

        else:
            # no data in public emb or just use esti_input
            full_input = torch.cat([esti_input.cpu()], dim=0)
            full_label = torch.cat([labels], dim=0)
        # print(full_label)

        full_input = full_input.to(self.device)
        full_label = full_label.to(self.device)
        return full_input, full_label

    def get_full_input_label2(self, esti_input, labels, attr, is_pu=False, is_full=True):
        if is_pu:
            full_input = copy.deepcopy(self.public_uemb)
        else:
            full_input = copy.deepcopy(self.public_uiemb)
        
        if len(full_input) > 0 and is_full:
            # pub data
            full_input = torch.cat(full_input, dim=0)
            idx = np.random.choice(full_input.shape[0], int(full_input.shape[0] * self.config['pubdata_ratio']), replace=False)
            full_input = full_input[idx] if idx.shape[0] > 0 else full_input
            full_label = copy.deepcopy(self.public_y[attr])
            full_label = torch.cat(full_label, dim=0)
            full_label = full_label[idx] if idx.shape[0] > 0 else full_label

            local_data = esti_input.cpu()
            idx_local = np.random.choice(local_data.shape[0], int(local_data.shape[0] * self.config['localdata_ratio']), replace=False)
            local_data = local_data[idx_local] if idx_local.shape[0] > 0 else local_data
            local_label = labels[idx_local] if idx_local.shape[0] > 0 else labels

            full_input = torch.cat([full_input, local_data], dim=0)            
            full_label = torch.cat([full_label, local_label], dim=0)
        else:
            # no data in public emb or just use esti_input
            full_input = torch.cat([esti_input.cpu()], dim=0)
            full_label = torch.cat([labels], dim=0)

        full_input = full_input.to(self.device)
        full_label = full_label.to(self.device)
        return full_input, full_label

    def fed_train_estimator(self, model_client, optimizer_esti, esti_input, privacy_labels, is_pu=False, is_full=True):
        # train estimator locally (use the client's data)
        for ratio, attr in zip(self.privacy_ratios, self.privacy_keys):
            esti_loss = 0
            for epoch in range(self.config['pries_epoch']):
                label = privacy_labels[attr]
                labels = self.label2torch(attr, esti_input, label)
                
                # esti_input, labels = self.get_full_input_label(esti_input, labels, attr, is_pu, is_full)
                esti_input, labels = self.get_full_input_label2(esti_input, labels, attr, is_pu, is_full)
                estimator, criterion = self.get_estimator_criterion(model_client, attr, is_pu)
                output = estimator(esti_input)
                esti_loss = criterion(output.squeeze(1), labels)

                optimizer_esti.zero_grad()
                esti_loss.backward()
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.)
                # ldp_add_noise(estimator, 10, 5, self.device)
                optimizer_esti.step()

    def fed_train_estimator_global(self, optimizer_esti, is_pu=False):
        # train estimator globally (train the estimator on trust party)
        for ratio, attr in zip(self.privacy_ratios, self.privacy_keys):
            esti_loss = 0
            for epoch in range(self.config['pries_epoch']):
                if is_pu:
                    full_input = copy.deepcopy(self.public_uemb)
                else:
                    full_input = copy.deepcopy(self.public_uiemb)
            
                if len(full_input) > 0:
                    # public emb 有数据
                    full_input = torch.cat(full_input, dim=0)

                    full_label = copy.deepcopy(self.public_y[attr])
                    full_label = torch.cat(full_label, dim=0)
                
                esti_input = full_input.to(self.device)
                labels = full_label.to(self.device)
                estimator, criterion = self.get_estimator_criterion(self.trust_party, attr, is_pu)
                output = estimator(esti_input)
                esti_loss = criterion(output.squeeze(1), labels)

                optimizer_esti.zero_grad()
                esti_loss.backward()
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.)
                # ldp_add_noise(estimator, 10, 5, self.device)
                optimizer_esti.step()

    def fed_estimator_inference(self, model_client, esti_input, privacy_labels, is_pu=False):
        esti_loss = 0
        for ratio, attr in zip(self.privacy_ratios, self.privacy_keys):
            label = privacy_labels[attr]
            labels = self.label2torch(attr, esti_input, label)
            estimator, criterion = self.get_estimator_criterion(model_client, attr, is_pu)
            output = estimator(esti_input)
            esti_loss += ratio * criterion(output.squeeze(1), labels)
        return esti_loss

    def min_mi(self, model_client, items, pos_mask, privacy_labels):
        # min MI(eu, item; P)
        p_input = model_client.get_train_input(items, pos_mask, is_item=True, is_eu=True, is_pu=False)
        loss = -1 * self.fed_estimator_inference(model_client, p_input, privacy_labels, is_pu=False) # 可以修改成与uniform distribution的kl loss 
        return loss
    
    def max_mi(self, model_client, items, privacy_labels):
        # max MI(pu)
        p_input = model_client.get_train_input(items, None, is_item=False, is_eu=False, is_pu=True)
        loss = self.fed_estimator_inference(model_client, p_input, privacy_labels, is_pu=True)
        return loss
    
    def min_H(self, model_client, privacy_labels):
        criterion = torch.nn.MSELoss()
        pri_emb = model_client.get_privacy_emb(privacy_labels)
        pu_emb = model_client.embedding_puser(torch.tensor([0]).to(self.device))
        loss = criterion(pri_emb, pu_emb)
        return loss

    def fed_train_single_batch(self, round_id, model_client, batch_data, optimizers, privacy_labels, pos_items):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            items, ratings = items.to(self.device), ratings.to(self.device)
        #     reg_item_embedding = reg_item_embedding.to(self.device)
        optimizer, _, optimizer_pri = optimizers
        pos_mask = ratings > 1e-5

        # update rec model.
        ratings_pred = model_client(items, pos_items)
        loss = self.client_crit(ratings_pred.view(-1), ratings)
        
        # if not self.config['pretrain']:
        if round_id > 0:
            if self.config['lam_eu'] > 0:
                # min MI(eu,item;P)
                if sum(pos_mask) > 0:
                    exclusive_loss = self.min_mi(model_client, items, pos_mask, privacy_labels)
                    loss += self.config['lam_eu'] * exclusive_loss
            if self.config['lam_pu'] > 0:
                # max MI(pu;P)
                FP_loss = self.max_mi(model_client, items, privacy_labels)
                # min H(pu|P)
                IP_loss = self.min_H(model_client, privacy_labels)
                loss += self.config['lam_pu'] * (FP_loss+IP_loss)
                
        optimizer.zero_grad()
        optimizer_pri.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_pri.step()    
        return model_client, loss.item()

    def aggregate_public_data(self, client_param, privacy_labels, num_part, uidx, user):
        if uidx == 0:
            self.tmp_public_uemb = []
            self.tmp_public_uiemb = []
            self.tmp_public_y = {key:[] for key in self.privacy_keys}
        if user in self.pri_user:
            ui_emb = []
            u_emb = copy.deepcopy(client_param['embedding_puser.weight']).cpu()
            ui_emb.append(copy.deepcopy(client_param['embedding_euser.weight']).cpu())
            ui_emb.append(copy.deepcopy(client_param[self.config['ITEM_NAME']]).cpu())
            u_emb_flat = torch.cat([u_emb])
            ui_emb_flat = torch.cat([grad.view(-1) for grad in ui_emb if grad is not None])
            self.tmp_public_uemb.append(u_emb_flat)
            self.tmp_public_uiemb.append(ui_emb_flat.reshape(1, -1))
            for attr in self.privacy_keys:
                label = privacy_labels[attr]
                labels = self.label2torch(attr, u_emb_flat, label)
                self.tmp_public_y[attr].append(labels)

            # print(self.tmp_public_uemb)
        if uidx == num_part - 1:
            self.public_uemb = self.tmp_public_uemb
            self.public_uiemb = self.tmp_public_uiemb
            self.public_y = self.tmp_public_y

    def aggregate_clients_params_peruser(self, client_param, num_part, t):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        if t == 0:
            self.tmp_server_model_param = {}
            # self.server_model_param = copy.deepcopy(client_param)
            for key in self.server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key] = user_params
        else:
            # print(self.server_model_param)
            for key in self.server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key].data += user_params
        
        if t == num_part - 1:
            self.server_model_param = {}
            # 最后一个client聚合后，再把聚合好的结果保留为server_param
            for key in self.server_keys:
                self.server_model_param[key] = self.tmp_server_model_param[key].data / num_part

    # train estimator locally
    def fed_train_a_round(self, all_train_data, round_id):
        """[global] train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(self.config['num_users'], num_participants, replace=False) # from 0 to num_users-1
        else:
            participants = np.random.choice(self.config['num_users'], self.config['clients_sample_num'], replace=False)

        # load privacy labels
        if self.config["dataset"] == 'douban':
            user_attr = pd.read_csv(f'../data/{self.config["dataset"]}/user.dat', sep=' ', header=None, names=['uid','location'])
        else:
            user_attr = pd.read_csv(f'../data/{self.config["dataset"]}/users.dat')

        # print(max(participants), min(participants))
        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        all_loss = 0
        for uidx, user in enumerate(participants):
            model_client = copy.deepcopy(self.client_model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated parameters from server. (FedNCF will receive all parameters)
            
            if round_id == 0 and self.config['finetune']:
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)
            if round_id != 0:
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)
            
            # ================ [Defining optimizers] =================================
            # optimizer for privacy estimators
            if self.config['dataset'] == 'ml-100k' or self.config['dataset'] == 'ml-1m' or self.config['dataset'] == 'ali-ads':
                optimizer_esti = torch.optim.SGD([
                    {'params': model_client.age_estimator.parameters()},
                    {'params': model_client.gen_estimator.parameters()},
                    {'params': model_client.occ_estimator.parameters()},
                    {'params': model_client.pu_age_estimator.parameters()},
                    {'params': model_client.pu_gen_estimator.parameters()},
                    {'params': model_client.pu_occ_estimator.parameters()}],
                lr=self.config['pries_lr'], momentum=0.9)
                # optimizer for privacy attribute encoder
                optimizer_pri = torch.optim.SGD([
                    {'params': model_client.embedding_age.parameters()},
                    {'params': model_client.embedding_gen.parameters()},
                    {'params': model_client.embedding_occ.parameters()}],
                lr=self.config['pries_lr'], momentum=0.9)

            # optimizer is responsible for updating score function.
            uemb_lr = self.config['lr_client'] / self.config['clients_sample_ratio'] * self.config['lr_eta'] 
            iemb_lr = self.config['lr_client'] * self.config['num_items'] * self.config['lr_eta'] 
            optimizer = torch.optim.SGD([
                    {'params': list(model_client.fc_layers.parameters()) + list(model_client.affine_output.parameters()), 'lr': self.config['lr_client']},
                    {'params': list(model_client.embedding_puser.parameters()) + list(model_client.embedding_euser.parameters()), 'lr': uemb_lr},
                    {'params': model_client.embedding_item.parameters(), 'lr': iemb_lr}
                ])
            
            optimizers = [optimizer, optimizer_esti, optimizer_pri]

            # load current user's training data and instance a train loader.        
            privacy_labels = {key:user_attr[key][user] for key in self.privacy_keys}
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update privacy estimator. [local]
            # train euser+item estimator
            ratings = torch.FloatTensor(all_train_data[2][user])
            items = torch.LongTensor(all_train_data[1][user]).to(self.device)
            pos_mask = ratings > 1e-5
            pos_items = items[pos_mask]
            # if self.config['dataset']
            if (round_id + 1) % self.config['pri_esti_round'] == 0:
                if self.config['lam_eu'] > 0:
                    # train euser+item estimator
                    esti_input = model_client.get_ei_input(items, pos_mask)
                    self.fed_train_estimator(model_client, optimizer_esti, esti_input, privacy_labels)
                if self.config['lam_pu'] > 0:
                    # train puser estimator
                    esti_input = model_client.get_pu_input(items)
                    self.fed_train_estimator(model_client, optimizer_esti, esti_input, privacy_labels, is_pu=True)

            epoch_loss = 0
            sample_num = 0
            # update client model. [local]
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_batch = self.fed_train_single_batch(round_id, model_client, batch, optimizers, privacy_labels, pos_items)
                    epoch_loss += loss_batch * len(batch[0])
                    sample_num += len(batch[0])

            all_loss += epoch_loss / sample_num
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # parameters client -> client
            # store client models' local parameters for personalization. 
            self.client_model_params[user] = {}
            for key in self.client_keys:
                self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()

            self.client_model_params[user][self.config['EUSER_NAME']] = copy.deepcopy(client_param['embedding_euser.weight'].data).cpu()
            avg_item_emb = torch.mean(client_param['embedding_item.weight'].data[items][pos_mask], dim=0)
            self.client_model_params[user][self.config['ITEM_NAME']] = copy.deepcopy(avg_item_emb).cpu()
            client_param[self.config['ITEM_NAME']] = copy.deepcopy(avg_item_emb).cpu()  # privacy-insensitive user will upload this params

            # parameters client -> server
            # store client models' local parameters for global update.
            # save u-i local graph for evaluation
            self.client_model_params[user]['pos_items'] = copy.deepcopy(pos_items).cpu()
            # if self.config['finetune']:
            self.aggregate_public_data(client_param, privacy_labels, len(participants), uidx, user)
            self.aggregate_clients_params_peruser(client_param, len(participants), uidx)
        
        return all_loss / len(participants)

    def fed_train_a_round_glob_esti(self, all_train_data, round_id):
        """[global] train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(self.config['num_users'], num_participants, replace=False) # from 0 to num_users-1
        else:
            participants = np.random.choice(self.config['num_users'], self.config['clients_sample_num'], replace=False)

        # load privacy labels
        if self.config["dataset"] == 'douban':
            user_attr = pd.read_csv(f'../data/{self.config["dataset"]}/user.dat', sep=' ', header=None, names=['uid','location'])
        else:
            user_attr = pd.read_csv(f'../data/{self.config["dataset"]}/users.dat')

        # print(max(participants), min(participants))
        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        all_loss = 0
        for uidx, user in enumerate(participants):
            model_client = copy.deepcopy(self.client_model)

            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated parameters from server. (FedNCF will receive all parameters)
            if round_id == 0 and self.config['finetune']:
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)
            if round_id != 0:
                # load rec model from server
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)
                # load estimators from trust party
                self.load_estimators(model_client)
            
            # ================ [Defining optimizers] =================================
            # optimizer for privacy estimators
            if self.config['dataset'] == 'ml-100k' or self.config['dataset'] == 'ml-1m' or self.config['dataset'] == 'ali-ads':
                optimizer_esti = torch.optim.SGD([
                    {'params': model_client.age_estimator.parameters()},
                    {'params': model_client.gen_estimator.parameters()},
                    {'params': model_client.occ_estimator.parameters()},
                    {'params': model_client.pu_age_estimator.parameters()},
                    {'params': model_client.pu_gen_estimator.parameters()},
                    {'params': model_client.pu_occ_estimator.parameters()}],
                lr=self.config['pries_lr'], momentum=0.9)
                # optimizer for privacy attribute encoder
                optimizer_pri = torch.optim.SGD([
                    {'params': model_client.embedding_age.parameters()},
                    {'params': model_client.embedding_gen.parameters()},
                    {'params': model_client.embedding_occ.parameters()}],
                lr=self.config['pries_lr'], momentum=0.9)

            # optimizer is responsible for updating score function.
            uemb_lr = self.config['lr_client'] / self.config['clients_sample_ratio'] * self.config['lr_eta'] 
            iemb_lr = self.config['lr_client'] * self.config['num_items'] * self.config['lr_eta'] 
            optimizer = torch.optim.SGD([
                    {'params': list(model_client.fc_layers.parameters()) + list(model_client.affine_output.parameters()), 'lr': self.config['lr_client']},
                    {'params': list(model_client.embedding_puser.parameters()) + list(model_client.embedding_euser.parameters()), 'lr': uemb_lr},
                    {'params': model_client.embedding_item.parameters(), 'lr': iemb_lr}
                ])
            
            optimizers = [optimizer, optimizer_esti, optimizer_pri]

            # load current user's training data and instance a train loader.        
            privacy_labels = {key:user_attr[key][user] for key in self.privacy_keys}
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update privacy estimator. [local]
            # train euser+item estimator
            ratings = torch.FloatTensor(all_train_data[2][user])
            items = torch.LongTensor(all_train_data[1][user]).to(self.device)
            pos_mask = ratings > 1e-5
            pos_items = items[pos_mask]
            # if self.config['dataset']
            # if (round_id + 1) % self.config['pri_esti_round'] == 0:
            if self.config['lam_eu'] > 0:
                # train euser+item estimator
                esti_input = model_client.get_ei_input(items, pos_mask)
                self.fed_train_estimator(model_client, optimizer_esti, esti_input, privacy_labels, is_full=False)
            if self.config['lam_pu'] > 0:
                # train puser estimator
                esti_input = model_client.get_pu_input(items)
                self.fed_train_estimator(model_client, optimizer_esti, esti_input, privacy_labels, is_pu=True, is_full=False)

            epoch_loss = 0
            sample_num = 0
            # update client model. [local]
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_batch = self.fed_train_single_batch(round_id, model_client, batch, optimizers, privacy_labels, pos_items)
                    epoch_loss += loss_batch * len(batch[0])
                    sample_num += len(batch[0])

            all_loss += epoch_loss / sample_num
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # parameters client -> client
            # store client models' local parameters for personalization. 
            self.client_model_params[user] = {}
            for key in self.client_keys:
                self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()
            
            self.client_model_params[user][self.config['EUSER_NAME']] = copy.deepcopy(client_param['embedding_euser.weight'].data).cpu()
            avg_item_emb = torch.mean(client_param['embedding_item.weight'].data[items][pos_mask], dim=0)
            self.client_model_params[user][self.config['ITEM_NAME']] = copy.deepcopy(avg_item_emb).cpu()
            client_param[self.config['ITEM_NAME']] = copy.deepcopy(avg_item_emb).cpu()

            # parameters client -> server
            # store client models' local parameters for global update.
            # save u-i local graph for evaluation
            self.client_model_params[user]['pos_items'] = copy.deepcopy(pos_items).cpu()
            # if self.config['finetune']:
            self.aggregate_public_data(client_param, privacy_labels, len(participants), uidx, user)
            self.aggregate_clients_params_peruser(client_param, len(participants), uidx)
        
        # train estimators on trust party
        optimizer_esti = torch.optim.SGD([
            {'params': self.trust_party.age_estimator.parameters()},
            {'params': self.trust_party.gen_estimator.parameters()},
            {'params': self.trust_party.occ_estimator.parameters()},
            {'params': self.trust_party.pu_age_estimator.parameters()},
            {'params': self.trust_party.pu_gen_estimator.parameters()},
            {'params': self.trust_party.pu_occ_estimator.parameters()}],
        lr=self.config['pries_lr'], momentum=0.9)

        if self.config['lam_eu'] > 0:
            # train euser+item estimator
            self.fed_train_estimator_global(optimizer_esti)
        if self.config['lam_pu'] > 0:
            # train puser estimator
            self.fed_train_estimator_global(optimizer_esti, is_pu=True)
        
        # print(self.trust_party.age_estimator.state_dict())
        return all_loss / len(participants)

    def load_local_param(self, user):
        user_param_dict = copy.deepcopy(self.client_model.state_dict())
        if not self.config['finetune']:
            # load global param from server
            for key in self.server_keys:
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).to(self.device)
            # load client local param
            if user in self.client_model_params.keys():
                for key in self.client_keys:
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)

        else:
            # if self.config['NAME'] == 'pretrain-apdf2':
            for key in self.server_keys:
                if key in self.server_model_param.keys():
                    user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).to(self.device)
            if user in self.client_model_params.keys():
                for key in self.client_keys:
                    if key in self.client_model_params[user].keys() and self.client_model_params[user][key].shape == user_param_dict[key].shape:
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(self.device)

        return user_param_dict

    def load_estimators(self, client_model):
        if self.config['dataset'] == 'ml-100k' or self.config['dataset'] == 'ml-1m' or self.config['dataset'] == 'ali-ads':
            age_esti_param = copy.deepcopy(self.trust_party.age_estimator.state_dict())
            client_model.age_estimator.load_state_dict(age_esti_param)
            gen_esti_param = copy.deepcopy(self.trust_party.gen_estimator.state_dict())
            client_model.gen_estimator.load_state_dict(gen_esti_param)
            occ_esti_param = copy.deepcopy(self.trust_party.occ_estimator.state_dict())
            client_model.occ_estimator.load_state_dict(occ_esti_param)

            pu_age_esti_param = copy.deepcopy(self.trust_party.pu_age_estimator.state_dict())
            client_model.pu_age_estimator.load_state_dict(pu_age_esti_param)
            pu_gen_esti_param = copy.deepcopy(self.trust_party.pu_gen_estimator.state_dict())
            client_model.pu_gen_estimator.load_state_dict(pu_gen_esti_param)
            pu_occ_esti_param = copy.deepcopy(self.trust_party.pu_occ_estimator.state_dict())
            client_model.pu_occ_estimator.load_state_dict(pu_occ_esti_param)
    
    def fed_evaluate(self, evaluate_data):
        """evaluate all client models' performance using testing data."""
        y = torch.FloatTensor([1]+[0]*self.config['NUM_NEG'])
        _, test_items = evaluate_data[0], evaluate_data[1]
        _, negative_items = evaluate_data[2], evaluate_data[3]
        if self.config['use_cuda'] is True:
            test_items = test_items.to(self.device)
            negative_items = negative_items.to(self.device)
            y = y.to(self.device)

        # store all users' test & negative item prediction score.
        test_scores, negative_scores = None, None
        # obtain items' prediction for each user.
        # user_ids = evaluate_data['uid'].unique()
        all_loss = 0
        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = self.load_local_param(user)
            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            with torch.no_grad():
                # obtain user's positive test information.
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_item = negative_items[user*self.config['NUM_NEG']: (user+1)*self.config['NUM_NEG']]
                # perform model prediction.
                pos_items = self.client_model_params[user]['pos_items']
                test_score = user_model(test_item, pos_items)
                # test_score = user_model(test_item)
                negative_score = user_model(negative_item, pos_items)
                y_hat = torch.cat((test_score, negative_score))
                loss = self.client_crit(y_hat.view(-1), y)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
            all_loss += loss.item()

        test_scores, negative_scores = test_scores.cpu(), negative_scores.cpu()
        # compute the evaluation metrics.
        recall, ndcg = compute_metrics(evaluate_data, test_scores, negative_scores, self.config['recall_k'])
        return recall, ndcg, all_loss/self.config['num_users']
    
    def get_params(self):
        save_params = {
            'server': copy.deepcopy(self.server_model_param),
            'client': copy.deepcopy(self.client_model_params)
        }
        return save_params

    def run_experiment(self, config, sample_generator):
        vali_recalls, vali_ndcgs = [], []
        test_recalls, test_ndcgs = [], []
        best_recall, best_ndcg, final_test_round, best_param = 0, 0, 0, None
        validate_data = sample_generator.validate_data
        test_data = sample_generator.test_data
        self.tmp_eu = self.config['lam_eu'] 
        self.tmp_pu = self.config['lam_pu'] 

        if config['finetune']:
            path = f'./saved_model/{config["dataset"]}/{config["NAME"]}'
            print(path)
            all_param = torch.load(path)
            self.client_model_params = copy.deepcopy(all_param['client'])
            self.server_model_param = copy.deepcopy(all_param['server'])
            logging.info('-' * 80)
            logging.info('Testing load param!')
            # test_recall, test_ndcg, test_loss = self.fed_evaluate(test_data)
            # logging.info(result2str('Recall', config['recall_k'], test_recall))
            # logging.info(result2str('NDCG', config['recall_k'], test_ndcg))
            # logging.info('Tst_Loss={:.5f}'.format(test_loss))
        
        for round in range(config['num_round']):
            logging.info('-' * 80)
            logging.info('-' * 80)
            logging.info('Round {} starts !'.format(round))

            logging.info('-' * 80)
            st_time = time.time()
            logging.info('Training phase!') # 每一个epoch都要重新sample
            if self.config['pretrain']:
                if round <= 3:
                    # 前期先预训练一下emb，先不训练estimator
                    self.config['lam_eu'] = 0
                    self.config['lam_pu'] = 0
                else:
                    self.config['lam_eu'] = self.tmp_eu
                    self.config['lam_pu'] = self.tmp_pu
            logging.info('lam_eu:{}, lam_pu:{}'.format(self.config['lam_eu'], self.config['lam_pu']))

            all_train_data = sample_generator.store_all_train_data(config['num_negative'])
            if self.config['is_esti_local']:
                train_loss = self.fed_train_a_round(all_train_data, round) # train estimator locally. this function will be slow
            # else:
            #     train_loss = self.fed_train_a_round_glob_esti(all_train_data, round) # train estimator globally
            
            ed_time = time.time()
            logging.info('Trn_time={:.4f} s'.format(ed_time - st_time))
            logging.info('Trn_Loss={:.5f}'.format(train_loss))

            logging.info('-' * 80)
            logging.info('Testing phase!')
            st_time = time.time()
            test_recall, test_ndcg, test_loss = self.fed_evaluate(test_data)
            ed_time = time.time()
            logging.info('Tst_time={:.4f} s'.format(ed_time - st_time))

            logging.info(result2str('Recall', config['recall_k'], test_recall))
            logging.info(result2str('NDCG', config['recall_k'], test_ndcg))
            logging.info('Tst_Loss={:.5f}'.format(test_loss))
            test_recalls.append(test_recall)
            test_ndcgs.append(test_ndcg)

            logging.info('-' * 80)
            logging.info('Validating phase!')
            vali_recall, vali_ndcg, vali_loss = self.fed_evaluate(validate_data)
            logging.info(result2str('Recall', config['recall_k'], vali_recall))
            logging.info(result2str('NDCG', config['recall_k'], vali_ndcg))
            logging.info('Val_Loss={:.5f}'.format(vali_loss))
            vali_recalls.append(vali_recall)
            vali_ndcgs.append(vali_ndcg)

            now_param = self.get_params()
            eval_all_privacy(config, now_param, None, self.client_keys, self.server_keys, self.client_model)
            logging.info('')
            
            if np.sum(vali_recall) >= np.sum(best_recall):
                best_recall = vali_recall
                best_ndcg = vali_ndcg
                final_test_round = round
                cnt = 0
                
            else:
                cnt += 1
                logging.info(f'Early stop at: {cnt} out of {config["earlystop"]}')
                if cnt >= config['earlystop']:
                    break

            if self.config['save_model']:
                    torch.save(now_param, f'./saved_model/{self.config["dataset"]}/{self.config["save_name"]}+lr{self.config["lr_client"]}+eta{self.config["lr_eta"]}')

        return test_recalls, test_ndcgs, final_test_round


class FedTrainer(Trainer):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        # client 自己根据config来判断是lightgcn还是ncf
        self.client_model = Client(config)

        self.device = config['device']
        self.dataset = config['dataset']
        self.client_model.to(self.device)
        self.mlp_keys = [k for k in self.client_model.state_dict().keys() if k.split('.')[0] in ['fc_layers', 'affine_output']]
        if self.dataset == 'ml-100k' or self.dataset == 'ali-ads' or self.dataset == 'ml-1m':
            estimators = [self.client_model.age_estimator, self.client_model.gen_estimator, self.client_model.occ_estimator]
            estimators += [self.client_model.pu_age_estimator, self.client_model.pu_gen_estimator, self.client_model.pu_occ_estimator]
            # estimators += [self.client_model.general]
            estimators_names = ['age_estimator', 'gen_estimator', 'occ_estimator']
            estimators_names += ['pu_age_estimator', 'pu_gen_estimator', 'pu_occ_estimator']
            # estimators_names += ['general']
            self.pestimator_keys = ['{}.{}'.format(name, k) for name, estimator in zip(estimators_names, estimators) for k in estimator.state_dict().keys()]
            # self.pestimator_keys_server = ['{}.{}'.format(name, k) for name, estimator in zip(estimators_names, estimators) for k in estimator.state_dict().keys() if k.split('.')[1] != '3']
            # self.pestimator_keys_client = ['{}.{}'.format(name, k) for name, estimator in zip(estimators_names, estimators) for k in estimator.state_dict().keys() if k.split('.')[1] == '3']
            self.priemb_keys = ['embedding_age.weight', 'embedding_gen.weight', 'embedding_occ.weight']
            
        # self.server_keys = ['embedding_item.weight', 'embedding_euser.weight'] # param saved in server
        # self.client_keys = ['embedding_puser.weight'] + self.mlp_keys + self.pestimator_keys + self.priemb_keys # param saved in client

        # self.client_keys = ['embedding_puser.weight'] + self.mlp_keys  # param saved in client
        # # both puser&euser are kept in client is better
        
        self.server_keys = ['embedding_item.weight'] # param saved in server
        self.client_keys = ['embedding_puser.weight', 'embedding_euser.weight'] + self.mlp_keys + self.pestimator_keys + self.priemb_keys 
        
        logging.info('server param: {}'.format(self.server_keys))
        logging.info('client param: {}'.format(self.client_keys))
        logging.info('client model: {}'.format(self.client_model))
        # exit(0)
        super(FedTrainer, self).__init__(config)


# baseline: UC-FedRec
class UC_FedTrainer(Trainer):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        # client 自己根据config来判断是lightgcn还是ncf
        self.client_model = Client(config)

        self.device = config['device']
        self.dataset = config['dataset']
        self.client_model.to(self.device)
        self.mlp_keys = [k for k in self.client_model.state_dict().keys() if k.split('.')[0] in ['fc_layers', 'affine_output']]
        if self.dataset == 'ml-100k' or self.dataset == 'ali-ads' or self.dataset == 'ml-1m':
            estimators = [self.client_model.age_estimator, self.client_model.gen_estimator, self.client_model.occ_estimator]
            estimators_names = ['age_estimator', 'gen_estimator', 'occ_estimator']
            self.pestimator_keys = ['{}.{}'.format(name, k) for name, estimator in zip(estimators_names, estimators) for k in estimator.state_dict().keys()]

        self.server_keys = ['embedding_item.weight', 'embedding_user.weight'] # param saved in server
        self.client_keys = ['embedding_user.weight'] + self.mlp_keys + self.pestimator_keys # param saved in client
        
        logging.info('server param: {}'.format(self.server_keys))
        logging.info('client param: {}'.format(self.client_keys))
        logging.info('client model: {}'.format(self.client_model))
        # exit(0)
        super(UC_FedTrainer, self).__init__(config)


                