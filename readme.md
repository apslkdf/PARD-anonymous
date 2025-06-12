### Run 
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
