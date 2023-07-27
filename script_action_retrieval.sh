test_name=$1

CUDA_VISIBLE_DEVICES=0 python action_retrieval.py \
  --lr 0.03 \
  --batch-size 512 \
  --knn-neighbours 1 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_0450.pth.tar \
  --finetune-dataset ntu120 --protocol cross_subject