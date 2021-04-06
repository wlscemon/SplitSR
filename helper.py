import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data
import os
import cv2
import torch


def test_network(net, trainloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    print('dsada')
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.show()

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


def collate_fn(self, data):
    imgs_list, boxes_list, classes_list = zip(*data)
    assert len(imgs_list) == len(boxes_list) == len(classes_list)
    batch_size = len(boxes_list)
    pad_imgs_list = []
    pad_boxes_list = []
    pad_classes_list = []

    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):
        img = imgs_list[i]
        pad_imgs_list.append(
            torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))

    max_num = 0
    for i in range(batch_size):
        n = boxes_list[i].shape[0]
        if n > max_num: max_num = n
    for i in range(batch_size):
        pad_boxes_list.append(
            torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
        pad_classes_list.append(
            torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

    batch_boxes = torch.stack(pad_boxes_list)
    batch_classes = torch.stack(pad_classes_list)
    batch_imgs = torch.stack(pad_imgs_list)

    return batch_imgs, batch_boxes, batch_classes


class DubbleDataSet(data.Dataset):
    """
    适用于4x，10x,20x超分辨，一次性读取对应的三组图
    """

    def __init__(self, image_4x_path, image_10x_path, image_name_list, crop=False, sample=None, brightness=None):

        self.image_4x_path = image_4x_path
        self.image_10x_path = image_10x_path
        # self.image_20x_path = image_20x_path
        # self.name_log_path = name_log_path
        self.crop = crop
        self.brightness = brightness
        self.image_name_list = image_name_list

        # init some utils function

    #     self.__read_name_log()
    #     if sample:
    #         random.seed(0)
    #         random.shuffle(self.image_name_list)
    #         self.image_name_list = self.image_name_list[:int(len(self.image_name_list)*sample)]
    # def __read_name_log(self):
    #     with open(self.name_log_path,'r') as f:
    #         for line in f:
    #             name = line.strip()
    #             self.image_name_list.append(name)
    #
    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, id):
        img_4x_path = os.path.join(self.image_4x_path, self.image_name_list[id])+'.png'
        # print(img_4x_path)
        img_10x_path = os.path.join(self.image_10x_path, self.image_name_list[id])+'x2.png'
        # print(img_10x_path)
        # img_20x_path = os.path.join(self.image_20x_path,self.image_name_list[id])

        img_4x = cv2.imread(img_4x_path)
        # print(img_4x.shape)
        img_10x = cv2.imread(img_10x_path)
        # img_20x = cv2.imread(img_20x_path)

        # BGR to RGB
        # img_4x = cv2.cvtColor(img_4x, cv2.COLOR_BGR2RGB)
        # img_10x = cv2.cvtColor(img_10x, cv2.COLOR_BGR2RGB)
        # img_20x = cv2.cvtColor(img_20x,cv2.COLOR_BGR2RGB)
        if self.crop:
            # because input 4x is 256
            img_4x = img_4x[64:192, 64:192, :]
            img_10x = img_10x[128:384, 128:384, :]
            # img_20x = img_20x[256:768,256:768,:]
        if self.brightness is not None:
            img_4x = cv2.cvtColor(img_4x, cv2.COLOR_RGB2HSV)
            img_10x = cv2.cvtColor(img_10x, cv2.COLOR_RGB2HSV)
            # img_20x = cv2.cvtColor(img_20x, cv2.COLOR_RGB2HSV)

            img_4x[..., 2] = img_4x[..., 2] + self.brightness
            img_10x[..., 2] = img_10x[..., 2] + self.brightness
            # img_20x[..., 2] = img_20x[..., 2] + self.brightness

            img_4x = cv2.cvtColor(img_4x, cv2.COLOR_HSV2RGB)
            img_10x = cv2.cvtColor(img_10x, cv2.COLOR_HSV2RGB)
            # img_20x = cv2.cvtColor(img_20x, cv2.COLOR_HSV2RGB)
        #     print('4x',img_4x.shape)
        #     print('10x',img_10x.shape)
        #     print('20x',img_20x.shape)
        # # H*W*C to C*H*W
        # img_4x = np.transpose(img_4x, axes=(2, 0, 1)).astype(np.float32) / 255.
        # img_10x = np.transpose(img_10x, axes=(2, 0, 1)).astype(np.float32) / 255.
        # img_20x = np.transpose(img_20x, axes= (2,0,1)).astype(np.float32)/255.
        # numpy array to torch tensor

        img_4x = torch.from_numpy(img_4x)
        # print(img_4x.shape)
        img_10x = torch.from_numpy(img_10x)
        # img_20x = torch.from_numpy(img_20x)

        return img_4x, img_10x
