import torch
import numpy as np

def asy_focal_SupConLoss(features, labels=None, mask=None, mixup=False, gamma=0, eta=0, temp=0.07):
    """
    Original code for SupConLoss: https://github.com/HobbitLong/SupContrast.
    Original code for focal_SupConLoss: https://github.com/Vanint/Core-tuning.
    """
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    temperature=temp
    base_temperature=temp
    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        if mixup:
            if labels.size(1)>1:
                weight_index = 10**np.arange(args.num_classes)  
                weight_index = torch.tensor(weight_index).unsqueeze(1).to("cuda")
                labels_ = labels.mm(weight_index.float()).squeeze(1)
                labels_ = labels_.detach().cpu().numpy()
                le = preprocessing.LabelEncoder()
                le.fit(labels_)
                labels = le.transform(labels_)
                labels=torch.unsqueeze(torch.tensor(labels),1)
        labels = labels.contiguous().view(-1, 1) 
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)
   
    anchor_feature = features.float()  
    contrast_feature = features.float()
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)  
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast
    logits_mask = torch.scatter(
        torch.ones_like(mask),  
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask   

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
    
    # compute weight
    weight = (1-torch.exp(log_prob)) ** gamma
    
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (weight * mask * log_prob).mean(1)

    # loss over positive
    mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
    mean_log_prob_pos = mean_log_prob_pos.view(batch_size)

    # compute reverse mask
    reverse_mask = (1 - mask) * logits_mask

    # compute negative probability
    prob_neg = 1 - torch.exp(log_prob)
    prob_neg.fill_diagonal_(1)

    # compute mean of log-likelihood over negative
    mean_log_prob_neg = (reverse_mask * torch.log(prob_neg)).mean(1)

    # loss over negative
    mean_log_prob_neg = - (temperature / base_temperature) * mean_log_prob_neg
    mean_log_prob_neg = mean_log_prob_neg.view(batch_size)

    # total loss
    loss = (mean_log_prob_pos + eta * mean_log_prob_neg).mean()

    if torch.isnan(loss):
         print("nan contrastive loss")
         loss=torch.zeros(1).to(device)          
    return loss
