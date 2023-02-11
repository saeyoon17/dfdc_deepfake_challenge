#!/bin/bash

ROOT_DIR='/input/dfdc-small-dataset/small_dataset'
NUM_GPUS='2'
OPT=$1
BATCH_SIZE=$2
LR=$3
WD=$4
echo $OPT
echo $BATCH_SIZE
echo $LR
echo $WD

cat ./dfdc_deepfake_challenge/configs/b7.json | jq .optimizer.type=$1
cat ./dfdc_deepfake_challenge/configs/b7.json | jq .optimizer.batch_size=$2
cat ./dfdc_deepfake_challenge/configs/b7.json | jq .optimizer.learning_rate=$3
cat ./dfdc_deepfake_challenge/configs/b7.json | jq .optimizer.weight_decay=$4

python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9902 training/pipelines/train_classifier.py \
 --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv   --fold 0 --seed 111 --data-dir $ROOT_DIR --prefix b7_111_ > logs/b7_111_sweep_test

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 555 --data-dir $ROOT_DIR --prefix b7_555_ > logs/b7_555

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 777 --data-dir $ROOT_DIR --prefix b7_777_ > logs/b7_777

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 888 --data-dir $ROOT_DIR --prefix b7_888_ > logs/b7_888

# python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 training/pipelines/train_classifier.py \
#  --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv  --fold 0 --seed 999 --data-dir $ROOT_DIR --prefix b7_999_ > logs/b7_999
