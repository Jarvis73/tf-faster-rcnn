#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
NUM_DETS=$4
THRESH_NMS=$5
THRESH_MAP=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  liver_ql)
    TRAIN_IMDB="liverQL_2016_train+liverQL_2017_train+liverQL_2018_train"
    TEST_IMDB="liverQL_2017_test+liverQL_2018_test"
    ITERS=350000
    ANCHORS="[4,8,12,16,24]"
    RATIOS="[0.5,0.7,1,1.5,2]"
    ;; 
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --num_dets ${NUM_DETS} \
    --thresh_nms ${THRESH_NMS} \
    --thresh_map ${THRESH_MAP} \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --num_dets ${NUM_DETS} \
    --thresh_nms ${THRESH_NMS} \
    --thresh_map ${THRESH_MAP} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

