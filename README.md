# tf-faster-rcnn for liver detection

**Acknowledge:** This reporitory was cloned from an earlier version of endernewton's tf-faster-rcnn([link](https://github.com/endernewton/tf-faster-rcnn)). For convenience I will change a lot of parts to apply my liver detection task.

## Still updating ...

## Change log

#### 2018-4-3

* Modify `./experiment/scripts/train_faster_rcnn.sh & test_faster_rcnn.sh`
  * add liver_ql case
  * `ANCHORS` [4,8,16]
  * `TRAIN_IMDB` two datasets (2016+2017)
  * `STEPSIZE` 70000
  * `ITERS` 100000

* Use `Leaky ReLU` in vgg16 and rpn

* Add image normalization: [-1, 1]
