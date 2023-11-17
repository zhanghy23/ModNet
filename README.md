# ModNet
## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)

## Project Preparation

#### A. Data Preparation

The dataset can be generated by function 'channel_generate' in 'main_ber_test_full.mat'.
You can download it from [Baidu Netdisk](https://pan.baidu.com/s/1NJJGpIs8G5RwCZvuEis73A?pwd=35ez). It is noted that testing dataset in the disk represents the validation dataset in our paper. Dataset are trained under v=360km/h.
The details of data setting can be found in our paper.

#### B. Checkpoints Downloading

The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1uawkDsRA2vhjLootAaeWJA?pwd=bipi).

#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
ModNet # The cloned ModNet repository
├── dataset
├── models
├── utils
├── main.py
├── ModNet Dataset  # The data folder
│   ├── H_train_360.mat
│   ├── ...
├── checkpoints  # The checkpoints folder
│   ├── best_loss_p1.pth
│   ├── ...
...
```

## Train ModNet from Scratch

An example is listed below. It will start advanced ModNet training from scratch. 
You will first train ModNet in the phase I with `--phase 1`.

``` 
python /home/ModNet/main.py \
  --data-dir '/home/ModNet Dataset' \
  --epochs 400 \
  --batch-size 100 \
  --workers 0 \
  --phase 1
  --gpu 0 \
```

After that, you can train ModNet in the phase II with `--phase 2`.

``` 
python /home/ModNet/main.py \
  --data-dir '/home/ModNet Dataset' \
  --epochs 400 \
  --batch-size 100 \
  --workers 0 \
  --phase 2
  --gpu 0 \
```
## Results and Reproduction

When ModNet has been trained, you can use 'test.py' to get the modulation/demodulation matrices from the validation dataset.

``` 
python /home/ModNet/test.py \

```

Then we provide 'main_ber_test_full.mat' to unify the modulation/demodulation matrices and test the BER of our modem structure compared with OFDM.


**To reproduce all these results, simple add `--evaluate` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` 
python /home/ModNet/main.py \
  --data-dir '/home/ModNet Dataset' \
  --pretrained './checkpoints/best_loss_p1.pth.pth' \
  --evaluate \
  --batch-size 100 \
  --workers 0 \
  --phase 1 #or 2\
  --cpu \
```
