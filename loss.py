import torch
from torch import nn



class SplitLoss(nn.Module):
    def __init__(self):
        super(SplitLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        self.L1_loss = nn.L1Loss()
        # self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.L1_loss(out_images, target_images)
        # TV Loss
        # tv_loss = self.tv_loss(out_images)
        return image_loss




if __name__ == "__main__":
    loss = SplitLoss()
    print(loss)
