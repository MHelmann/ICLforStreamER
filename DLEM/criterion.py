import torch
import torch.nn.functional as F


def info_nce_loss(device, features, batch_size, n_views, temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels


def loss_cal_ICL_old(old_x, old_x_aug, new_x, alpha):
    """
    Copied from https://github.com/RingBDStack/ICL-Incremental-InfoNCE/blob/main/criterion.py
    """
    T = 0.1
    batch_size, _ = old_x.size()
    
    old2new_sim_matrix = torch.einsum('ik,jk->ij', old_x, new_x) / torch.einsum('i,j->ij', old_x.norm(dim=1), new_x.norm(dim=1))
    old2new_sim_matrix = torch.exp(old2new_sim_matrix / T)
    
    old2old_sim_matrix = torch.einsum('ik,jk->ij', old_x, old_x_aug) / torch.einsum('i,j->ij', old_x.norm(dim=1), old_x_aug.norm(dim=1))
    old2old_sim_matrix = torch.exp(old2old_sim_matrix / T)
    
    pos_sim = old2old_sim_matrix[range(batch_size), range(batch_size)]
       
    loss = (1-alpha) + alpha * (pos_sim + old2new_sim_matrix.sum(dim=1)) / (old2old_sim_matrix.sum(dim=1))
    loss = torch.log(loss).mean()

    return loss


def loss_cal_ICL_new(new_x, new_x_aug, old_x, new_num):
    """
    Copied from https://github.com/RingBDStack/ICL-Incremental-InfoNCE/blob/main/criterion.py
    """
    T = 0.1
    batch_size, _ = new_x.size()

    new2new_sim_matrix = torch.einsum('ik,jk->ij', new_x, new_x_aug) / torch.einsum('i,j->ij', new_x.norm(dim=1), new_x_aug.norm(dim=1))
    new2new_sim_matrix = torch.exp(new2new_sim_matrix / T)
    
    pos_sim = new2new_sim_matrix[range(batch_size), range(batch_size)]
    
    if old_x is None:
        new2old_sim_matrix = torch.zeros([batch_size, 1]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        new2old_sim_matrix = torch.einsum('ik,jk->ij', new_x, old_x) / torch.einsum('i,j->ij', new_x.norm(dim=1), old_x.norm(dim=1))
        new2old_sim_matrix = torch.exp(new2old_sim_matrix / T)
        
    loss = pos_sim / (torch.cat((new2new_sim_matrix[:new_num, new_num], pos_sim[new_num:])) + new2new_sim_matrix[:, :new_num].sum(dim=1) + new2old_sim_matrix.sum(dim=1))
    loss = - torch.log(loss).mean()

    return loss