# hw_unet

## Instruction for `infer.py`:

In your kaggle notebook, first have the dataset `bkai-igh-neopolyp` ready.
And use the configuration of 2x Tesla T4 GPUs

Do these before cloning the repo:

```
!pip install segmentation-models-pytorch
!pip install wget
```

`cd` to the cloned folder and run `infer.py`
There is 1 argument for the file: Choose your `path` to save the pretrained model, or let it save by default.

`output.csv` is in `working` folder


