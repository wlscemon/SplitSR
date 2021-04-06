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
from data_utils import TrainDatasetFromFolder, display_transform, ValDatasetFromFolder
from myhelper import update_lr
from emaill import mail

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Train Super Resolution Split Models')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')# split model also need to change manually
parser.add_argument('--rotate_degrees', default=90, type=int, help='random rotate degrees')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--learning_rate', default=1*1e-4, type=int, help='learning rate')
# parser.add_argument('--beta1', default=100, type=0.9, help='beta 1')
# parser.add_argument('--beta2', default=100, type=0.999, help='beta2')
parser.add_argument('--epsilon', default=1e-7, type=int, help='epsilon')
parser.add_argument('--model_name', default="mobile_log/checkLr_e1_100_lr1.0.pth", type=str, help='model epoch name')
parser.add_argument('--model_note', default="checkLr", type=str, help='comment of model pth')
parser.add_argument('--flow_check', default=False, type=str, help='set true to check flow of code')

if __name__ == '__main__':
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
    
    train_set = TrainDatasetFromFolder('../pre_data/high_2', upscale_factor=UPSCALE_FACTOR, degrees=ROTATE_DEGREES,
                                       crop_size=CROP_SIZE, low_dir = '../pre_data/low_2')
    #val_set = ValDatasetFromFolder('../pre_data/high_p', upscale_factor=UPSCALE_FACTOR)
    #val_set = ValDatasetFromFolder('../pre_data/high', upscale_factor=UPSCALE_FACTOR, degrees=ROTATE_DEGREES,
    #                               crop_size=CROP_SIZE, low_dir = '../pre_data/low')
    train_loader = DataLoader(dataset=train_set, num_workers=12, batch_size=3, shuffle=True)
    #val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    model = SplitSR()
    print(f"# num of train set is {len(train_set)}")
    
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    #model.load_state_dict(torch.load( MODEL_NAME))
    
    #match_val_array = np.loadtxt('../image_match/temp/val_array.txt', delimiter=',')

    
    criterion = SplitLoss()
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=EPS)
    
    results = {'loss': [], 'psnr': [], 'ssim': []}
    loss_list = []
    mail('alpha = 0.5, split training is begin with lr = 1, 50 epochs')
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == 2:
            mail('your split model is training successfully')
        #LR = update_lr(LR, loss_list)
        #print(LR)
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0}

        model.train()
        
        ontrain_num = -1
        for data, target in train_bar:
            #break
            ontrain_num  = ontrain_num + 1
            if FLOW_CHECK:
                if ontrain_num >= 10:
                    break
            #if match_val_array[ontrain_num] > THRESHOLD:
            #    continue
            
            batch_size = data.size(0)
            running_results['batch_sizes'] = running_results['batch_sizes'] + batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            #real_img = real_img*255
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            #z = z*255
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = model(z)
            
    
            # model.zero_grad()
            # real_out = netD(real_img).mean()
            # fake_out = netD(fake_img).mean()
            # d_loss = 1 - real_out + fake_out
            # d_loss.backward(retain_graph=True)
            # optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            model.zero_grad()
            #print(f"real: {real_img.shape}")
            #print(f"low: {data.shape}")
            #print(f"fake: {fake_img.shape}")
            loss = criterion(fake_img, real_img)
            torch.autograd.set_detect_anomaly(True)
            loss.backward()

            
            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()
            
            
            optimizer.step()

            # loss for current batch before optimization 
            running_results['loss'] += loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            # running_results['d_score'] += real_out.item() * batch_size
            # running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                epoch, NUM_EPOCHS, running_results['loss']* 255 / running_results['batch_sizes']))
            model_name = 'mobile_log/' + str(MODEL_NOTE) + '_e' + str(epoch) + '_' + str(NUM_EPOCHS) + '_lr' + str(LR*1e4) + '.pth'
            torch.save(model.state_dict(), model_name) 
        loss_list.append(running_results['loss'] / running_results['batch_sizes'])
            #with open('mobile_log/model_list.txt', 'w') as f:
            #   temp = str('\n' + model_name)
            #   f.write(temp) 
np.savetxt("mobile_log/loss_list_new.txt", loss_list,fmt='%f',delimiter=',') 


ret = mail('your alpha= 0.5 split training model is done.')
'''
        model.eval()
        out_path = 'training_results/bio_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                #print("it can't be seen, bug in the previous code!")
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr*255
                hr = val_hr
                #val_hr_restore = val_hr_restore*255
                #print(hr[0][:5])
                #print(lr[0][:5])
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = model(lr)
                sr = sr/255
                #print(sr[0][:5])
        
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
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
               
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
   
        # save model parameters
        
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

