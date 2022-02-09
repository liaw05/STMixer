# MAE pretrained model

The pretrained model of vit base is from [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch), a unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

## download MAE pretrained model
You can download the pretrained model from [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch):
|   model  | pretrain | finetune | accuracy | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|
| vit-base |   400e   |   100e   |   83.1%  | [Google drive](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing) |

To get deit type weight, you should run:
```bash
python mae_convert_to_deit.py
```

## download MAE-deit pretrained model
You can alse directly download the converted model from [baidu yun](https://pan.baidu.com/s/1FhV7L4yGzYVLO8bunNZ5Jw), Extraction code: maed.
