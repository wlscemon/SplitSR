import numpy as np

def update_lr(lr, loss_list):
#update learning rate if loss is steady
    scale = 1
    
    #if len(loss_list) >= 3:
    #    test_loss = loss_list[-3:]
    #    mean_loss = np.mean(test_loss)
    #    loss_range = max(test_loss) - min(test_loss)
    #    if loss_range <= 3 and mean_loss >=25:
    #        scale = scale * 0.3
    new_lr = lr * scale
    return new_lr
    
       
   
