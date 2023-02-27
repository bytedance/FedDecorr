device_id=0
MODEL=mobilenetv2
ALPHA=0.5
PARTITION=iid
WD=1e-5

CUDA_VISIBLE_DEVICES=$device_id python3 main.py \
    --dataset=cifar10 \
    --model=$MODEL \
    --approach=fedoptim \
    --lr=0.1 \
    --weight_decay=$WD \
    --epochs=1 \
    --n_comm_round=100 \
    --n_parties=10 \
    --partition=$PARTITION \
    --alpha=$ALPHA \
    --logdir='./logs/' \
    --datadir='./data/' \
    --ckptdir='./models/' \
    --server_optimizer=adagrad \
    --server_momentum=0.9 \
    --server_momentum_second=0.99 \
    --server_learning_rate=0.01

