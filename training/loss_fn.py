import torch.nn as nn
import torch

ce_loss = nn.CrossEntropyLoss()
def ce_maskloss(output, target, mask):
    '''
    Cross entropy loss with mask. If in a batch the data length is different, the data could be padded and a mask will be passed along with 1 indicate invalid data
    In METRE, the tasks use data of the same length, so this function is not needed
    '''
    loss = [ce_loss(output[i][mask[i]==0].mean(dim=-2).unsqueeze(0), target[i]) for i in range(len(output))]
    return torch.mean(torch.stack(loss))