### PARD
source code of the paper "Privacy-Aware Representation Decoupling in Federated Recommendation against Attribute Inference Attacks"

### Abstract
Federated recommender systems (FedRec) aim to preserve user privacy by keeping sensitive data on client devices and sharing only model parameters with a central server. However, FedRec is still vulnerable to attribute inference attacks (AIA), where server-side adversaries exploit uploaded parameters to infer users’ private attributes. Existing approaches face a suboptimal privacy-performance trade-off. Privacy-focused methods mask attribute-related features in representations to protect sensitive information, but degrade recommendation accuracy. In contrast, performance-focused methods preserve accuracy by retaining these features but risk privacy leakage through uploaded representations. To balance privacy and performance, we propose PARD, a privacy-aware representation decoupling framework that explicitly decouples representations into privacy-relevant and privacy-irrelevant components. Only the privacy-irrelevant part is uploaded to the server, and the privacy-relevant part is retained locally. We introduce mutual information (MI) objectives to realize the decoupling: (1) minimizing MI between privacy-irrelevant representations and sensitive attributes to suppress leakage, and (2) maximizing MI for privacy-relevant representations to retain personalized preference signals. Since exact MI computation is intractable, we derive variational bounds and estimate them using privacy estimators under adversarial and cooperative training paradigms. Experimental results demonstrate that PARD outperforms state-of-the-art methods in both recommendation accuracy and privacy preservation. We will release the source code after the acceptance of the paper.

### Start 
#### (A) end2end train 
- pretrain=True, finetune=False；lam_eu=0.5，lam_pu=0.5；
```
python run_grid_search.py --GNN [False] or [True] --num_round 200 --dataset ml-1m --earlystop 10 --pretrain True --finetune False # GNN=False for applying pard to fedncf; GNN=true for applying it to fedgnn;
```

#### (B) pretrain+finetune
-The advantage of this is that,
    -Pretrain the model to achieve a better recommendation result; When reusing estimator fine-tuning, the recommendation effect will not be too poor;
    -Because adversarial training takes a lot of time (as currently all clients need to train the estimator first, which is too slow), pre training would result in much faster training speed;
-The downside of end2end training is that during adversarial training, the privacy loss of the estimator may be too easy to optimize, causing the optimizer to train the privacy loss instead of the recloss, resulting in the recommendation model falling into sub_optimal mode;

```
python run_experiment.py --GNN [False] or [True] --num_round xx --dataset ml-1m --earlystop 10 --pretrain True --finetune False --save_model True --save_name xxx.pkl
python run_experiment.py --GNN [False] or [True] --num_round yy --dataset ml-1m --earlystop 10 --pretrain False --finetune True
```
for example,
python run_experiment.py --num_round 45 --save_model True --save_name APDF-pretran-ml1m.pkl --device_id 1 --dataset ml-1m --GNN False --earlystop 10 --pretrain True --finetune False
