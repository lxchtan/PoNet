SUFFIX=PoNet_bookcourpus_wikipedia_dupe5
LOGNAME=`date +%Y%m%d%H`_${SUFFIX}.log
OUTPUT=outputs/${SUFFIX}
MODEL_PATH=chtan/ponet-base-uncased

if [ ! -d logs ]; then
  mkdir logs
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 31030 run_pretrained.py \
    --config_name ${MODEL_PATH} \
    --tokenizer_name ${MODEL_PATH} \
    --dataset_name bookcorpus \
    --dataset2_name wikipedia \
    --dataset2_config_name 20200501.en \
    --label_names labels next_sentence_label \
    --save_total_limit 5 \
    --dupe_factor 5 \
    --num_train_epochs 500 \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --evaluation_strategy steps \
    --save_steps 5000 \
    --eval_steps 5000 \
    --logging_dir ${OUTPUT} \
    --report_to tensorboard \
    --do_train \
    --do_eval \
    --ignore_data_skip \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --sharded_ddp simple \
    --output_dir ${OUTPUT} > logs/${LOGNAME} 2>&1 &