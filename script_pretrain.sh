test_name=$1


if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}

fi
 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  --nproc_per_node=4 pretrain.py   \
  --lr 0.0005   --batch-size 1024  \
  --checkpoint-path ./checkpoints/${test_name} \
  --schedule 351  --epochs 451  \
  --pre-dataset ntu120  --protocol cross_subject | tee -a ./checkpoints/${test_name}/train.log