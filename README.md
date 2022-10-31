# ava

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-squaree&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat-square&logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat-square&logo=scipy&logoColor=%white)

implementation of preprint **Dynamic sky replacement and video harmonization** 
that automatically generate realistic and dramatic sky backgrounds in videos with controllable 
styles.

## paper
you can also check out the ava output paper, titled [Ava (Dynamic sky replacement and video harmonization)](https://github.com/slowy07/paper/blob/main/Ava.pdf?raw=true)\
from:
```
https://github.com/slowy07/paper/blob/main/Ava.pdf
```

## take ava for a spin !
### locally
you can clone the project and run it locally

download checkpoints_G_coord_resnet50 [here](https://drive.google.com/u/0/uc?id=1COMROzwR4R_7mym6DL9LXhHQlJmJaV0J&export=download)
```bash
git clone https://github.com/slowy07/ava
cd ava
pip install -r requirements.txt

# for district canyon
python skymagic.py --path ./config/config-canyon-district9ship.json

# for skymoon
python skymagic.py --path ./config/config-annarbor-supermoon.json
```

or you can use custom image by configuring the json configuration file like so:
```json
// config-custom-replacement.json
{
  "net_G": "coord_resnet50",
  "ckptdir": "./checkpoints_G_coord_resnet50",

  "input_mode": "video",
  // save custom video on `test_videos`
  "datadir": "./test_videos/custom_video.mp4",
  // save custom image for replacement on `skybox` folder
  "skybox": "custom_image.jpg",

  "in_size_w": 384,
  "in_size_h": 384,
  "out_size_w": 845,
  "out_size_h": 480,

  "skybox_center_crop": 0.5,
  "auto_light_matching": false,
  "relighting_factor": 0.8,
  "recoloring_factor": 0.5,
  "halo_effect": true,

  "output_dir": "./jpg_output",
  "save_jpgs": false
}
```
running with custom config by
```
python skymagic.py --path ./config/config-custom-replacement.json
```

### retrain sky matting model

download complete CVPRW20-SkyOpt dataset from google [here](https://github.com/google/sky-optimization).
We only uploaded a very small part of it due to the limited space of the repository. The mini-dataset we included in this repository is only used as an example to show how the structure of the file directory is organized.

```
unzip datasets.zip
python train.py \
	--dataset cvprw2020-ade20K-defg \
	--checkpoint_dir checkpoints \
	--vis_dir val_out \
	--in_size 384 \
	--max_num_epochs 200 \
	--lr 1e-4 \
	--batch_size 8 \
	--net_G coord_resnet50
```


### online
or run it online using google colab, you can check it out at [ava google colab](https://colab.research.google.com/drive/1Hgi09hung57bnNIun4L7n93LpEb-beLk?usp=sharing) or copy paste the link directly
```
https://colab.research.google.com/drive/1Hgi09hung57bnNIun4L7n93LpEb-beLk?usp=sharing
```
#### how to use

1. click the `connect` button\
  ![connect](.github/connect.png)
2. then run it by opening the `runtime` tab, then click `run all`\
  ![runAll](.github/run.png)
3. wait for result


## our device requirements

our testing so far do in online and offline testing

**online testing**

- Nvidia GPU Server TESLA K80
- 12 GB Rams

**offline testing and running including maintance**

im testing on my personal computer which test on device:

- 16 core of processor Ryzen 9 Model 100-100000059WOF 5900x
- 32 GB of Ram Model F4-4000C17D-32GTZRB Z 4000 Mhz 
- Geforce RTX 3090 24GB

## output result

![output of demo castle render](./output/demo-castle.gif?raw=true)
![output of demo supermoon](./output/demo-supermoon.gif)
![output of demo canyon](./output/demo-canyon.gif)

## performance

speed performance of our method at different output
resolution and the time spent in different processing phases
1. sky matting
2. motion estimation
3. blending

| resoulution | speed per fps (read) | phase 1 | phase 2 | phase 3 |
| :--------: | :------------------: | :------------------: | :------------------: | :------------------: |
| 640×360 | 98.03 | 0.0235 | 0.0070 | 0.0070 |
| 854×480 | 87.92 | 0.0334 | 0.0150 | 0.0186 |
| 1280×720 | 68.04 | 0.0565 | 0.0329 | 0.0386 |

The inference speeds are tested on a desktop PC with an NVIDIA
Geforce RTX 3090 GPU card and an 16 core of processor AMD Ryzen
9 Model 5900x. The speed at different output resolution and
time spent in different processing stages are recorded. We can
see our method reaches a real-time processing speed (98 fps) at
the output resolution 640 x 320 and a near real-time processing
speed (87 fps) at 854x480 but still has large rooms for speed
up. As there is a considerable part of the time spent in the
sky matting stage, one may easily speed up the processing
pipeline by replacing the ResNet-50 with a more efficient CNN
backbone, e.g MobileNet or EfficientNet.

## limitation

The limitation of our method is twofold. First, since our sky matting network is only trained on daytime images, our method may fail to detect the sky regions on nighttime videos. Second, when there are no sky pixels during a certain period of time in a video, or there are no textures in the sky, the motion of the sky background cannot be accurately modeled.

### Donate this project

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/arfyslowy)
