test_name=$1

CUDA_VISIBLE_DEVICES=0 python action_recognition.py \
  --lr 0.03 \
  --batch-size 512 \
  --pretrained  ./checkpoints/${test_name}/checkpoint_0450.pth.tar \
  --finetune-dataset ntu120 --protocol cross_subject
