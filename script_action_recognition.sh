dir=$1
dataset=$2
protocol=$3


CUDA_VISIBLE_DEVICES=0 python action_recognition.py \
  --lr 0.03 \
  --batch-size 512 \
  --pretrained  ./checkpoints/${dir}/checkpoint_0450.pth.tar \
  --finetune-dataset ${dataset} --protocol ${protocol}
