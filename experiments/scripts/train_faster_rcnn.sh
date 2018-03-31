#!/bin/bash

set -x  # 执行指令后，会先显示该指令及参数
set -e  # 若指令传回值不等于0，则立即退出shell

export PYTHONUNBUFFERED="True"

# 命令行参数
GPU_ID=$1
DATASET=$2
NET=$3
# MED_IMG True
array=( $@ )                          # 参数列表
len=${#array[@]}                      # 获取数组长度
EXTRA_ARGS=${array[@]:3:$len}         # 其他参数列表
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}    # 把参数中的空格替换为下划线

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  liver_ql)
    TRAIN_IMDB="liverQL_2018_train"
    TEST_IMDB="liverQL_2018_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")      # 定义同时输出到stdout和文件 "$LOG"
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then    # 如果字符串的长度不为0
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then      # 如果检查点文件不存在
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  fi
fi

# ./experiments/scripts/test_faster_rcnn.sh $@
