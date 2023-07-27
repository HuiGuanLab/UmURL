cuda_device=$1
test_name=$2


if [ ! -d ./checkpoints/${test_name} ];then
    mkdir -p ./checkpoints/${test_name}_xsub

fi
 CUDA_VISIBLE_DEVICES=1,2 torchrun  --nproc_per_node=2 pretrain.py   \
  --lr 0.0005   --batch-size 1024  \
  --checkpoint-path ./checkpoints/${test_name}_xsub \
  --schedule 351  --epochs 451  \
  --pre-dataset ntu120  --protocol cross_subject | tee -a ./checkpoints/${test_name}_xsub/train.log