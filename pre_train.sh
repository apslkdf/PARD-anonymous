is_GNN=True
# is_GNN=False
# dataset='ml-100k'
dataset='ali-ads'
# dataset='ml-1m'
# pre_name='APDF-FedNCF-pretrain15-ml100k-v2.pkl'
# pre_name='APDF-FedNCF-pretrain15-ml100k.pkl+lr0.5+eta80'
# pre_name='apdf-pre2-titan.pkl'
# pre_name='APDF-FedGNN-pretrain-epoch30.pkl'
pre_name='APDF-FedGNN-pretrain-epoch30.pkl'
# pre_name='APDF-FedNCF-pretrain13-ml1m.pkl+lr0.5+eta80'
ratio=1.
is_esti_local=True
device='1'
# device='0'

python run_experiment.py --GNN $is_GNN --num_round 30 --save_model True --save_name $pre_name --dataset $dataset --earlystop 10 --pretrain True --finetune False --is_esti_local $is_esti_local --device_id $device