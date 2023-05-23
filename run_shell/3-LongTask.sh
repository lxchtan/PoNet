MODEL=chtan/ponet-base-uncased
OUTPRE=`pwd`

cal(){
  NAME=`date +%Y%m%d%H`_${MAINTASK}_ep${num_train_epochs}_bz$((bz*GAS))_lr${lr}
  OUTPUT=outputs/${MAINTASK}/${NAME}

  CUDA_VISIBLE_DEVICES=${GPUID} torchrun --nproc_per_node=${GPU_NUMS} run_long_classification.py \
    --model_name_or_path ${MODEL} \
    --task_name ${MAINTASK} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 4096 \
    --gradient_accumulation_steps ${GAS} \
    --per_device_train_batch_size ${bz} \
    --per_device_eval_batch_size ${bz} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_steps 500 \
    --eval_steps 500 \
    --save_steps 5000 \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --save_strategy no \
    --report_to tensorboard \
    --fp16 \
    --logging_dir ${OUTPUT} \
    --output_dir ${OUTPUT}  > logs/${MAINTASK}/${NAME}.log 2>&1
}

search(){
  num_train_epochs=${NEP}
  if [ ! -d logs/${MAINTASK} ]; then
    mkdir -p logs/${MAINTASK}
  fi
  lr=3e-5; cal
  lr=5e-5; cal
}

GPUID=0,1
GPU_NUMS=2
GAS=1
bz=16
### ------- Arxiv-11 --------
MAINTASK=arxiv; NEP=10; search;
### ------- IMDb --------
# MAINTASK=imdb; NEP=10; search;
### ------- HND --------
# MAINTASK=hnd; NEP=10; search;
### ------- Yelp-5 --------
# MAINTASK=yelp; NEP=2; search;
