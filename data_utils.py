from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, RandomHorizontalFlip, RandomRotation


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(degrees, crop_size):
    return Compose([
        #RandomCrop(crop_size),
        #RandomHorizontalFlip(),
        #RandomRotation(degrees),
        ToTensor(),
    ])


def train_lr_transform(upscale_factor, crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        #CenterCrop(400),
        ToTensor()
    ])

def down_size(upscale_factor):
    return Compose([
        ToPILImage(),
        #Resize([1024, 1024], interpolation=Image.BICUBIC),
        Resize([512, 512], interpolation=Image.BICUBIC),
        ToTensor()
    ])

def low_path_f_high(h_path):
    devi = '_'
    h_split = h_path.split('_')
    low_path = devi.join(h_split[:3]) + '_low_' + h_split[-1]
    return low_path
    
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, degrees, crop_size, low_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.image_filenames_l = [join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)]
        #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(degrees, crop_size)
        self.lr_transform = train_lr_transform(upscale_factor, crop_size)
        self.down_size = down_size(upscale_factor)
        self.low_dir = low_dir

    def __getitem__(self, index):
        h_path = self.image_filenames[index]
        hr_image_origin = self.hr_transform(Image.open(h_path))
        devi = '_'
        h_split = h_path.split('_')
        low_path = devi.join(h_split[-5:-2]) + '_low_' + h_split[-1]
        low_path = low_path.split('/')
        low_path = low_path[-1]
        low_path = join(self.low_dir, low_path)
        lr_image = self.hr_transform(Image.open(low_path))
        hr_image = self.down_size(hr_image_origin)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)
        
class ValDatasetFromFolder(Dataset):

    def __init__(self, dataset_dir, upscale_factor, degrees, crop_size, low_dir):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        #self.image_filenames_l = [join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)]
        #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(degrees, crop_size)
        self.low_dir = low_dir
        self.lr_transform = train_lr_transform(upscale_factor, crop_size)
        self.down_size = down_size(upscale_factor)
        #self.find_low_path = low_path_f_high()#for place holder

    def __getitem__(self, index):
        h_path = self.image_filenames[index]
        hr_image_origin = self.hr_transform(Image.open(h_path))
        #print(h_path)
        #h_shape = hr_image_origin.shape[:2]
        #hr_image = Image.open(self.image_filenames[index])
        #print(self.low_dir)
        devi = '_'
        h_split = h_path.split('_')
        #print(h_split)
        #h_split = h_split.split('/')
        #h_split = h_split[-1]
        low_path = devi.join(h_split[-5:-2]) + '_low_' + h_split[-1]
        low_path = low_path.split('/')
        low_path = low_path[-1]
        #print(low_path)
        # above are to find the low path from high path
        #lo = self.low_dir
        #hi = self.find_low_path(h_path)
        #print(hi)
        low_path = join(self.low_dir, low_path)
        lr_image = self.hr_transform(Image.open(low_path))
        #print(l_path)
        #print(low_path)
        hr_image = self.down_size(hr_image_origin)
        #hr_image = hr_image*255
        # print(self.image_filenames)
        #lr_image = self.lr_transform(hr_image)
        #lr_image = lr_image*255
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)
'''
class ValDatasetFromFolder(Dataset):

     def __init__(self, dataset_dir, upscale_factor):
         super(ValDatasetFromFolder, self).__init__()
         self.upscale_factor = upscale_factor
         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
 
     def __getitem__(self, index):
         hr_image = Image.open(self.image_filenames[index])
         w, h = hr_image.size
         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
         hr_image = CenterCrop(crop_size)(hr_image)
         #print(hr_image.size)
         lr_image = lr_scale(hr_image)
         hr_restore_img = hr_scale(lr_image)
         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
 
     def __len__(self):
         return len(self.image_filenames)
'''

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
