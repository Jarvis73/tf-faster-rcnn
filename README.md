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

#### 2018-4-4
* Remove 'only keep anchors inside the image'

* STEP_SIZE = [30000, 70000]  

* RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N
* STEP_SIZE = [100000] ITERS = 150000

#### 2018-4-5
* vgg16.yml, RPN_BATCHSIZE = 512

* Add focal loss and decrease batch size for training  
  I have checked the signal/noise and found that the number of positive anchors is in the range [1, 20], maybe some more while the negative samples are about 10 thousand. So I have to restrict the batch size to 100(found in `./experiment/cfgs/vgg16.yml`).  
  Up to now, I have gotten a mean IoU of **0.480** with test instruction:
  ```bash
  bash experiments/scripts/test_faster_rcnn.sh 5 liver_ql vgg16 3 0.02
  ```

* Add abdominal mask as anchor filter. 

* Modify to two thresholds in test routine, `THRES_PRE_NMS` and `THRESH_POST_NMS`  
  Up to now, I have gotten a mean IoU of **0.575** with test instuction:
  ```bash
  bash experiments/scripts/test_faster_rcnn.sh 5 liver_ql vgg16 3 0. 0.015
  ``