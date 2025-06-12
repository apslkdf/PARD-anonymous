import torch
# from trainer import Trainer
# from utils import use_cuda
import numpy as np
from attacker import *
import logging
# from fedncf import Server


class LightGCN(torch.nn.Module):
    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def inference(self, pos_items, drop_prob=0.5):
        # # N_u = len(pos_items); N_i = 1;
        drop_mask = torch.rand(pos_items.size()) > drop_prob
        if sum(drop_mask) > 0:
            # avoid nan
            dropped_pos_items = pos_items[drop_mask]
        else:
            dropped_pos_items = pos_items
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        pos_item_emb = item_emb[dropped_pos_items]
        # new user emb
        new_user_emb = torch.mean(pos_item_emb, dim=0, keepdim=True)
        new_user_emb = (new_user_emb + user_emb) / 2
        # new item emb
        # new_item_emb = item_emb.clone()
        # for idx in pos_items:
        #     new_item_emb[idx] = item_emb[idx] + user_emb
        # return new_user_emb, new_item_emb
        return new_user_emb, item_emb
        

    def forward(self, item_indices, pos_items=None):
        # if pos_items:
        user_emb, item_emb = self.inference(pos_items, self.config['gnn_drop'])
        user_embedding = user_emb[torch.tensor([0] * len(item_indices)).cuda()]
        item_embedding = item_emb[item_indices]

        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.LeakyReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass

class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.dataset = config['dataset']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.device = config['device']

        # Privacy-Preserving user emb
        self.embedding_puser = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # Privacy-Exclusive user emb
        self.embedding_euser = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Score function
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        
        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if config['dataset'] == 'ml-100k' or config['dataset'] == 'ml-1m' or config['dataset'] == 'ali-ads':
        # g_{phi}: emb -> privacy attributes
            esti_dim = self.latent_dim * 2
            # euser+item estimator
            # self.general = General_Module(esti_dim)
            self.age_estimator = Discriminator(config, esti_dim, attr='age')
            self.gen_estimator = Discriminator(config, esti_dim, attr='gender')
            self.occ_estimator = Discriminator(config, esti_dim, attr='occupation')
            # puser estimator
            self.pu_age_estimator = Discriminator(config, self.latent_dim, attr='age')
            self.pu_gen_estimator = Discriminator(config, self.latent_dim, attr='gender')
            self.pu_occ_estimator = Discriminator(config, self.latent_dim, attr='occupation')
        # f_p: privacy attr -> attr emb
            self.embedding_age = torch.nn.Embedding(num_embeddings=config['num_age'], embedding_dim=self.latent_dim) 
            self.embedding_gen = torch.nn.Embedding(num_embeddings=config['num_gender'], embedding_dim=self.latent_dim) 
            self.embedding_occ = torch.nn.Embedding(num_embeddings=config['num_occupation'], embedding_dim=self.latent_dim) 

    def inference(self, pos_items, drop_prob=0.5):
        # # N_u = len(pos_items); N_i = 1;
        drop_mask = torch.rand(pos_items.size()) > drop_prob
        if sum(drop_mask) > 0:
            # avoid nan
            dropped_pos_items = pos_items[drop_mask]
        else:
            dropped_pos_items = pos_items
        puser_emb = self.embedding_puser.weight
        euser_emb = self.embedding_euser.weight
        item_emb = self.embedding_item.weight
        pos_item_emb = item_emb[dropped_pos_items]
        
        # new user emb
        new_puser_emb = torch.mean(pos_item_emb, dim=0, keepdim=True) 
        new_puser_emb = (new_puser_emb + puser_emb) / 2
        # new user emb
        new_euser_emb = torch.mean(pos_item_emb, dim=0, keepdim=True)
        new_euser_emb = (new_euser_emb + euser_emb) / 2

        return new_puser_emb, new_euser_emb, item_emb
        

    def forward(self, item_indices, pos_items=None):
        if self.config['GNN']:
            puser_emb, euser_emb, item_emb = self.inference(pos_items, self.config['gnn_drop'])
            puser_embedding = puser_emb[torch.tensor([0] * len(item_indices)).to(self.device)]
            euser_embedding = euser_emb[torch.tensor([0] * len(item_indices)).to(self.device)]
            item_embedding = item_emb[item_indices]
        else:
            puser_embedding = self.embedding_puser(torch.tensor([0] * len(item_indices)).to(self.device))
            euser_embedding = self.embedding_euser(torch.tensor([0] * len(item_indices)).to(self.device))
            item_embedding = self.embedding_item(item_indices)
        
        vector = torch.cat([euser_embedding + puser_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
    
    def get_privacy_emb(self, privacy_labels):
        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m' or self.dataset == 'ali-ads':
            age_idx = torch.tensor([privacy_labels['age']]).to(self.device)
            gen_idx = torch.tensor([privacy_labels['gender']]).to(self.device)
            occ_idx = torch.tensor([privacy_labels['occupation']]).to(self.device)
            age_embedding = self.embedding_age(age_idx)
            gen_embedding = self.embedding_gen(gen_idx)
            occ_embedding = self.embedding_occ(occ_idx)
            pri_vector = torch.cat([age_embedding, gen_embedding, occ_embedding], dim=0)
            return torch.mean(pri_vector, dim=0, keepdim=True)
        elif self.dataset == 'douban':
            loc_idx = torch.tensor([privacy_labels['location']]).to(self.device)
            loc_embedding = self.embedding_loc(loc_idx)
            return loc_embedding
        elif self.dataset == 'bookcrossing':
            age_idx = torch.tensor([privacy_labels['age']]).to(self.device)
            loc_idx = torch.tensor([privacy_labels['location']]).to(self.device)
            age_embedding = self.embedding_age(age_idx)
            loc_embedding = self.embedding_loc(loc_idx)
            pri_vector = torch.cat([age_embedding, loc_embedding], dim=0)
            return torch.mean(pri_vector, dim=0, keepdim=True)

    def get_item_emb(self, items, pos_masks):
        with torch.no_grad():
            emb_list = []
            if sum(pos_masks) > 0:
                item_embeddings = self.embedding_item(items)[pos_masks]
                item_embedding = torch.mean(item_embeddings, dim=0, keepdim=True)
                emb_list.append(item_embedding)
            return emb_list
    
    def get_input(self, items, pos_masks):
        with torch.no_grad():
            emb_list = []
            if sum(pos_masks) > 0:
                item_embeddings = self.embedding_item(items)[pos_masks]
                item_embedding = torch.mean(item_embeddings, dim=0, keepdim=True)
                emb_list.append(item_embedding)
            puser_embedding = self.embedding_puser(torch.tensor([0] * len(items)).to(self.device))
            euser_embedding = self.embedding_euser(torch.tensor([0] * len(items)).to(self.device))
            emb_list += [puser_embedding, euser_embedding]
            # emb_list += [euser_embedding]
            # print(item_embedding.shape, puser_embedding.shape, euser_embedding.shape)
            return torch.cat(emb_list, dim=0)
    
    def get_ei_input(self, items, pos_masks):
        with torch.no_grad():
            item_embeddings = torch.mean(self.embedding_item(items)[pos_masks], dim=0, keepdim=True)
            euser_embedding = self.embedding_euser(torch.tensor([0] * item_embeddings.shape[0]).to(self.device))
            # print(item_embeddings, euser_embedding)
            emb_list = [euser_embedding, item_embeddings]
            return torch.cat(emb_list, dim=1)

    def get_pu_input(self, items):
        with torch.no_grad():
            puser_embedding = self.embedding_puser(torch.tensor([0]).to(self.device))
            emb_list = [puser_embedding]
            return torch.cat(emb_list, dim=0)

    def get_train_input(self, items, pos_masks, is_item=True, is_eu=True, is_pu=True):
        # with torch.no_grad():
        if is_pu:
            puser_embedding = self.embedding_puser(torch.tensor([0] * len(items)).to(self.device))
            emb_list = [puser_embedding]
            return torch.cat(emb_list, dim=0)
        elif is_item and is_eu:
            item_embeddings = self.embedding_item(items)[pos_masks]
            euser_embedding = self.embedding_euser(torch.tensor([0] * item_embeddings.shape[0]).to(self.device))
            emb_list = [euser_embedding, item_embeddings]
            return torch.cat(emb_list, dim=1)

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class General_Client(torch.nn.Module):
    def __init__(self, config):
        super(General_Client, self).__init__()
        self.config = config
        self.dataset = config['dataset']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.device = config['device']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Score function
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        
        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if config['dataset'] == 'ml-100k' or config['dataset'] == 'ml-1m' or config['dataset'] == 'ali-ads':
        # g_{phi}: emb -> privacy attributes
            esti_dim = self.latent_dim * 2
            # euser+item estimator
            # self.general = General_Module(esti_dim)
            self.age_estimator = Discriminator(config, esti_dim, attr='age')
            self.gen_estimator = Discriminator(config, esti_dim, attr='gender')
            self.occ_estimator = Discriminator(config, esti_dim, attr='occupation')

    def inference(self, pos_items, drop_prob=0.5):
        # # N_u = len(pos_items); N_i = 1;
        drop_mask = torch.rand(pos_items.size()) > drop_prob
        if sum(drop_mask) > 0:
            # avoid nan
            dropped_pos_items = pos_items[drop_mask]
        else:
            dropped_pos_items = pos_items
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        pos_item_emb = item_emb[dropped_pos_items]
        
        # new user emb
        new_user_emb = torch.mean(pos_item_emb, dim=0, keepdim=True) 
        new_user_emb = (new_user_emb + user_emb) / 2

        return new_user_emb, item_emb
        

    def forward(self, item_indices, pos_items=None):
        if self.config['GNN']:
            user_emb, item_emb = self.inference(pos_items, self.config['gnn_drop'])
            user_embedding = user_emb[torch.tensor([0] * len(item_indices)).to(self.device)]
            item_embedding = item_emb[item_indices]
        else:
            user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).to(self.device))
            item_embedding = self.embedding_item(item_indices)
        
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating
    

    def get_item_emb(self, items, pos_masks):
        with torch.no_grad():
            emb_list = []
            if sum(pos_masks) > 0:
                item_embeddings = self.embedding_item(items)[pos_masks]
                item_embedding = torch.mean(item_embeddings, dim=0, keepdim=True)
                emb_list.append(item_embedding)
            return emb_list
    
    def get_input(self, items, pos_masks):
        with torch.no_grad():
            emb_list = []
            if sum(pos_masks) > 0:
                item_embeddings = self.embedding_item(items)[pos_masks]
                item_embedding = torch.mean(item_embeddings, dim=0, keepdim=True)
                emb_list.append(item_embedding)
            user_embedding = self.embedding_user(torch.tensor([0] * len(items)).to(self.device))
            emb_list += [user_embedding]
            return torch.cat(emb_list, dim=0)
    
    # def get_ei_input(self, items, pos_masks):
    #     with torch.no_grad():
    #         item_embeddings = torch.mean(self.embedding_item(items)[pos_masks], dim=0, keepdim=True)
    #         euser_embedding = self.embedding_euser(torch.tensor([0] * item_embeddings.shape[0]).to(self.device))
    #         # print(item_embeddings, euser_embedding)
    #         emb_list = [euser_embedding, item_embeddings]
    #         return torch.cat(emb_list, dim=1)

    # def get_pu_input(self, items):
    #     with torch.no_grad():
    #         puser_embedding = self.embedding_puser(torch.tensor([0]).to(self.device))
    #         emb_list = [puser_embedding]
    #         return torch.cat(emb_list, dim=0)

    def get_train_input(self, items, pos_masks):
        user_embedding = self.embedding_user(torch.tensor([0] * len(items)).to(self.device))
        emb_list = [user_embedding]
        return torch.cat(emb_list, dim=0)

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass
