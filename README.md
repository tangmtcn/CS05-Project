# Video Classifier

## Dataset

Place videos in the `dataset` folder in different subfolders by class.

We privode a dataset containing 7 classes of videos. It can be downloaded [here](https://cs05-vc.s3.amazonaws.com/dataset/tiktok-dataset.tar.gz).

You can also run the folloing script to download it.

```sh
wget https://cs05-vc.s3.amazonaws.com/dataset/tiktok-dataset.tar.gz -O dataset.tar.gz
tar -zxvf dataset.tar.gz
rm dataset.tar.gz
```

## Training

Install the requirements:

```sh
python3 -m pip install tensorflow>=2.1.0 matplotlib opencv-python
```

Run the training script

```sh
python3 train.py
```

## Use pre-trained model

A pre-trained model is provided to perform prediction.

```sh
cp -R model/pretrained_model model/model_save
```

## Predict

To classify a new video `video/a.mp4`, run the script:

```sh
python3 predict.py video/a.mp4
```
