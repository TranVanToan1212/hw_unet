
import torch
import os
import cv2

import pandas as pd
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode, Normalize
import albumentations as A
import argparse
from pathlib import Path
import gdown
import wget

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=Path, default='/kaggle/working/pretrained/unet_model.pth', help = 'model save path')

args = parser.parse_args()

# url = 
# wget.download(url, path)
# '/k
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_path = str(args.path)
pretrained_directory = '/'.join(pretrained_path.split('/')[:-1])
if not os.path.exists(pretrained_directory):
    os.mkdir(pretrained_directory)
# print(pretrained_path)
# [:-12]
# pretrained_path = '/kaggle/working/pretrained/unet_model.pth'
# '/kaggle/input/checkpoint_path/unet_model.pth'
# url = 'https://drive.google.com/file/d/1_ZLyCf6dl1XwY2CVUw7um75_BkaVYvf1/view?usp=sharing'
# wget.download(url, pretrained_path)
# pretrained_path = args.path
url = 'https://drive.google.com/uc?id=1_ZLyCf6dl1XwY2CVUw7um75_BkaVYvf1'
gdown.download(url, pretrained_path)

ENCODER = 'resnet152'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [0,1,2]

ACTIVATION = 'softmax2d'

# def load_model(model, optimizer, path):
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint["model"])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print('model loaded')
#     return model, optimizer

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

checkpoint = torch.load(pretrained_path)
new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

model = nn.DataParallel(model)
model.to(device)
# summary(model, (3, 224, 224))
print('model loaded, now inferencing...')

transform = Compose([
    Resize((224,224),
        interpolation = InterpolationMode.BICUBIC, 
        antialias = True),
    PILToTensor(),
])

class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform, 
            ):
        super(UNetTestDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        images_list = [images_path+i for i in images_list]
        
        self.images_list = images_list
        self.transform = transform        

    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        
        data = self.transform(data)
        data = data.type(torch.float32)
        data = Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(data)        

        return data, img_path, h, w
    
    def __len__(self):
        return len(self.images_list)        

path = '/kaggle/input/bkai-igh-neopolyp/test/test/'
unet_test_dataset = UNetTestDataClass(path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=8, shuffle=True)


model.eval()
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W
    
    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST_EXACT)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join("/kaggle/working/predicted_masks/", filename))

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

MASK_DIR_PATH = '/kaggle/working/predicted_masks' # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)        
