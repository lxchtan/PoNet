export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=0

MODEL=$1
OUTPUT=../logs

if [ ! -d ${OUTPUT} ]; then
  mkdir ${OUTPUT}
fi

python3 run_tasks.py --model ${MODEL} --task listops --output ${OUTPUT}
python3 run_tasks.py --model ${MODEL} --task text --output ${OUTPUT}
python3 run_tasks.py --model ${MODEL} --task retrieval --output ${OUTPUT}
python3 run_tasks.py --model ${MODEL} --task image --output ${OUTPUT}
python3 run_tasks.py --model ${MODEL} --task pathfinder32-curv_contour_length_14 --output ${OUTPUT}
