num_train_epochs=4
MODEL=outputs/ponet-base-uncased
OUTPRE=`pwd`

if [ ! -d logs/glue ]; then
  mkdir logs/glue
fi

cal_mlp(){
  NAME=`date +%Y%m%d%H`_${TASK_NAME}_ep${num_train_epochs}_bz$((bz*GAS))_lr${lr}
  OUTPUT=outputs/glue/${NAME}

  CUDA_VISIBLE_DEVICES=${GPUID} python -u run_glue.py \
    --model_name_or_path ${MODEL} \
    --overwrite_output_dir \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --per_device_train_batch_size ${bz} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --save_steps 5000 \
    --save_strategy no \
    --fp16 \
    --logging_dir ${OUTPUT} \
    --output_dir ${OUTPUT}  > logs/glue/${NAME}.log 2>&1
}


search(){
  GAS=1
  bz=128
  lr=3e-4; cal_mlp
  lr=1e-4; cal_mlp
  lr=5e-5; cal_mlp
  lr=3e-5; cal_mlp

  GAS=1
  bz=64
  lr=3e-4; cal_mlp
  lr=1e-4; cal_mlp
  lr=5e-5; cal_mlp
  lr=3e-5; cal_mlp

  GAS=1
  bz=32
  lr=3e-4; cal_mlp
  lr=1e-4; cal_mlp
  lr=5e-5; cal_mlp
  lr=3e-5; cal_mlp

  GAS=1
  bz=16
  lr=3e-4; cal_mlp
  lr=1e-4; cal_mlp
  lr=5e-5; cal_mlp
  lr=3e-5; cal_mlp

  GAS=1
  bz=8
  lr=3e-4; cal_mlp
  lr=1e-4; cal_mlp
  lr=5e-5; cal_mlp
  lr=3e-5; cal_mlp
}

GPUID=0; TASK_NAME=cola; search &
# GPUID=1; TASK_NAME=stsb; search &
# GPUID=2; TASK_NAME=mrpc; search &
# GPUID=3; TASK_NAME=rte; search &
# GPUID=4; TASK_NAME=sst2; search &
# GPUID=5; TASK_NAME=qqp; search &
# GPUID=6; TASK_NAME=qnli; search &
# GPUID=7; TASK_NAME=mnli; search &
