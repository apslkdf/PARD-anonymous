import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, config, embed_dim, attr):
        super(Discriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        # self.criterion = nn.NLLLoss()
        self.attr = attr
        self.dataset = config['dataset']

        # if attr == 'age':
        #     self.out_dim = config['num_age']
        #     self.activation = F.log_softmax
        #     self.drop = 0.5
        #     self.net = nn.Sequential(
        #     # nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=self.drop),
        #     nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )
        # elif attr == 'gender':
        #     if config['num_gender'] == 2:
        #         self.out_dim = 1
        #     else:
        #         self.out_dim = config['num_gender']
            
        #     self.activation = torch.sigmoid
        #     self.drop = 0.5
        #     self.net = nn.Sequential(
        #     # nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=self.drop),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=self.drop),
        #     nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )
        # elif attr == 'occupation':
        #     if config['num_occupation'] == 2:
        #         self.out_dim = 1
        #     else:
        #         self.out_dim = config['num_occupation']
            
        #     self.activation = F.log_softmax
        #     self.drop = 0.3
        #     self.net = nn.Sequential(
        #         # nn.BatchNorm1d(num_features=self.embed_dim),
        #         nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(p=self.drop),
        #         nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )

        
        if attr == 'age':
            self.out_dim = config['num_age']
            self.activation = F.log_softmax
            self.drop = 0.5
            self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )
        elif attr == 'gender':
            if config['num_gender'] == 2:
                self.out_dim = 1
            else:
                self.out_dim = config['num_gender']
            
            self.activation = torch.sigmoid
            self.drop = 0.5
            self.net = nn.Sequential(
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.02),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.02),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.02),
            nn.Dropout(p=self.drop),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

        elif attr == 'occupation':
            if config['num_occupation'] == 2:
                self.out_dim = 1
            else:
                self.out_dim = config['num_occupation']
            
            self.activation = F.log_softmax
            self.drop = 0.3
            self.net = nn.Sequential(
                # nn.BatchNorm1d(num_features=self.embed_dim),
                nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
                nn.LeakyReLU(0.2),
                nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
                nn.LeakyReLU(0.2),
                nn.Dropout(p=self.drop),
                nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )
        
    def forward(self, ents_emb):
        scores = self.net(ents_emb)
        if self.attr == 'gender':
            output = torch.sigmoid(scores)
        else:
            output = self.activation(scores, dim=1)
        return output

    def predict(self, ents_emb):
        with torch.no_grad():
            scores = self.net(ents_emb)
            if self.attr == 'gender':
                output = self.activation(scores)
                preds = (scores > torch.Tensor([0.5]).to(ents_emb.device)).float() * 1
            else:
                output = self.activation(scores, dim=1)
                preds = output.argmax(dim=1)  # get the index of the max
            return output.squeeze(), preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def get_param(self):
        param_dict = self.net.state_dict()
        grads = []
        # for name in param_dict.keys():
        if self.attr == 'gender':
            param = param_dict['6.weight']
        else:
            param = param_dict['5.weight']
        grads.append(param.data)

        flat_grad = torch.cat([grad.view(-1) for grad in grads if grad is not None])
        return flat_grad

    # def create_kl_divergence(self, u_g_embeddings):
    #     predict = self.forward(u_g_embeddings)
    #     target = (torch.ones(predict.shape) / predict.shape[1]).to(u_g_embeddings.device)
    #     loss = F.kl_div(predict, target, reduction='batchmean')
    #     return loss


# class General_Module(nn.Module):
#     def __init__(self, embed_dim):
#         super(General_Module, self).__init__()
#         self.embed_dim = int(embed_dim)
#         self.drop = 0.5
#         self.net = nn.Sequential(
#             # nn.BatchNorm1d(num_features=self.embed_dim),
#             nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=self.drop),
#             # nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
#             # nn.LeakyReLU(0.2),
#             # nn.Dropout(p=self.drop),
#             # nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
#         )
        
#     def forward(self, ents_emb):
#         return self.net(ents_emb)
         

