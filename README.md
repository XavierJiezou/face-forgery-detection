# Face-Forgery-Detection

[![Space](https://img.shields.io/badge/🤗HugginngFace-Space-orange)](https://huggingface.co/spaces/XavierJiezou/face-forgery-detection)
[![Dataset](https://img.shields.io/badge/🤗HugginngFace-Dataset-orange)](https://huggingface.co/datasets/XavierJiezou/FaceShield)

## Online Demo 

<a href="https://huggingface.co/spaces/XavierJiezou/face-forgery-detection"><img src="https://github.com/user-attachments/assets/a55e4bed-a249-4c71-b381-cac5129830d5" width="70%"></a>

## Prepare Dataset

<!--
- Input Modal
  - Text
  - Mask
  - Text & Mask

- Model Paradigm
  - GAN
  - Diffusion
  - GAN & Diffusion
  - Flow

- Data Augmentation
  - Mixup
  - Cutmix
  - Cutout
  - Mosaic
-->

```bash
data
|-- fake
|   |-- clip2latent
|   |-- collaborativediffusion
|   |-- controlnet
|   |-- e2style
|   |-- e3facenet
|   |-- flux.1-dev
|   |-- gcdp
|   |-- inade
|   |-- ipadapter
|   |-- pix2pix_turbo
|   |-- pixelfaceplus
|   |-- scgan
|   |-- semflow
|   |-- stable diffusion 1.5
|   |-- tedigan
|   |-- uniteandconquer
|   |-- face-mogle
|-- real
|   |-- celeba
|   |-- ffhq
```


## Supported Method

- [x] [CVPR 2024] [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [x] [WACV 2025] [Wavelet-CLIP](https://github.com/lalithbharadwajbaru/Wavelet-CLIP)
- [x] [CVPR 2025] [D³](https://github.com/BigAandSmallq/D3)
