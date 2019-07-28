# toynet
Toys for Talki


How to make Quality Image Dataset/Net


Labelling the Images

- Open the images folder in labelImg


Should the box contain all of the object, or a balanced box with equal part cut off and background included.

## Tensorflow Training

Tensorflow models git repository must be checked out at `/Volumes/Project/models`.

The Tensorflow retrain guide is focused on classification of the whole image.
https://www.tensorflow.org/hub/tutorials/image_retraining


How to train your own Object Detector with TensorFlow’s Object Detector API
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

Official local training guide
- Naming could be better
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md?source=post_page---------------------------


## Utils

App Store: RectLabel
- Seems to be better than imgLabel app

Microsoft VoTT
https://github.com/microsoft/VoTT/releases


https://github.com/tzutalin/ImageNet_Utils


python retrain.py --logtostderr --train_dir=./images/train --pipelin_config_path=./ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync.config



### Rotating the images based on EXIF info

https://superuser.com/questions/670818/how-to-automatically-rotate-images-based-on-exif-data
https://www.xnview.com/en/nconvert/#downloads

> nconvert -jpegtrans exif -overwrite images/apple/*.JPG


List of pretrained networks
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md




## Target Model for Talki

https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/3
The module contains a trained instance of the network, packaged to get feature vectors from images.


MODEL_DIR=/Volumes/Projects/toynet/ python object_detection/export_tflite_ssd_graph.py --pipeline_config_path ${MODEL_DIR}ssdlite_mobilenet_v2_coco_toynet.config --trained_checkpoint_prefix ${MODEL_DIR}ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt.data-00000-of-00001 --output_directory ${MODEL_DIR}export2


cd research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
PIPELINE_CONFIG_PATH=/Volumes/Projects/toynet/ssdlite_mobilenet_v2_coco_toynet.config
MODEL_DIR=/Volumes/Projects/toynet/
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr
cd /Volume/Projects/toynet
