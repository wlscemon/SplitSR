import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from splitsr import SplitSR
from PIL import ImageFile

from loss import SplitLoss
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from myhelper import update_lr
from emaill import mail
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Train Super Resolution Split Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--rotate_degrees', default=90, type=int, help='random rotate degrees')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--learning_rate', default=0.003*1e-4, type=int, help='learning rate')
# parser.add_argument('--beta1', default=100, type=0.9, help='beta 1')
# parser.add_argument('--beta2', default=100, type=0.999, help='beta2')
parser.add_argument('--epsilon', default=1e-7, type=int, help='epsilon')
parser.add_argument('--model_name', default="mobile_log/new_tune.pth", type=str, help='model epoch name')
parser.add_argument('--model_note', default="checkLr", type=str, help='comment of model pth')
parser.add_argument('--flow_check', default=False, type=str, help='set true to check flow of code')



if __name__ == '__main__':
    starttime = time.time()
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    ROTATE_DEGREES = opt.rotate_degrees
    LR = opt.learning_rate
    EPS = opt.epsilon
    MODEL_NAME = opt.model_name
    MODEL_NOTE = opt.model_note
    FLOW_CHECK = opt.flow_check
    THRESHOLD = 0.8
    #h_shape = [1024, 1024]
    
    #train_set = TrainDatasetFromFolder('../pre_data/high', upscale_factor=UPSCALE_FACTOR, degrees=ROTATE_DEGREES,
     #                                  crop_size=CROP_SIZE, low_dir = '../pre_data/low')

    val_set = ValDatasetFromFolder('../pre_data/high_2', upscale_factor=UPSCALE_FACTOR, degrees=ROTATE_DEGREES,
                                       crop_size=CROP_SIZE, low_dir = '../pre_data/low_2')
    #train_loader = DataLoader(dataset=train_set, num_workers=5, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    model = SplitSR()
    #print(f"# num of train set is {len(train_set)}")
    
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load( MODEL_NAME))
    
    #match_val_array = np.loadtxt('../image_match/temp/val_array.txt', delimiter=',')

    
    criterion = SplitLoss()
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=EPS)
    
    results = {'loss': [], 'psnr': [], 'ssim': []}
    loss_list = []

model.eval()
out_path = 'training_results/new_tune_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

with torch.no_grad():
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    val_num = 1
    for val_lr, val_hr in val_bar:
        val_num = val_num+1
        if val_num >= 10:
            break
        #print("it can't be seen, bug in the previous code!")
        #high and low image is 0-1
        #split model have a 255 images of input and output
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = val_lr
        hr = val_hr
        #print(hr[0][:1])
        #val_hr_restore = val_hr_restore*255
        #print(hr[0][:5])
        #print(lr[0][:5])
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = model(lr)
        #print(sr[0][:5])
        sr = sr
        

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_lr.data.cpu().squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
       
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 3)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + '_index_%d.png' % ( index), padding=5)
        index += 1
    endtime = time.time()
    print('总共的时间为:', round(endtime - starttime, 2),'secs')
# save model parameters
'''
# torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
# save loss\scores\psnr\ssim
results['loss'].append(running_results['loss'] / running_results['batch_sizes'])
# results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
# results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
# results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
results['psnr'].append(valing_results['psnr'])
results['ssim'].append(valing_results['ssim'])

if epoch % 10 == 0 and epoch != 0:
    out_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        index=range(1, epoch + 1))
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
'''

