# hw_unet

## Instruction for `infer.py`:

In your kaggle notebook, first have the dataset `bkai-igh-neopolyp` ready.
And use the configuration of 2x Tesla T4 GPUs.

Do these before cloning the repo:

```
!pip install segmentation-models-pytorch
!pip install gdown
```

`clone` from `git`, `cd`, and run `infer.py` (here I'm assuming the cloned folder name is also `hw_unet`):

```
!git init
!git clone 'https://github.com/TranVanToan1212/hw_unet.git' '/kaggle/working/hw_unet'
%cd /kaggle/working/hw_unet
!python infer.py
```

`output.csv` is in `/working/hw_unet` folder.


