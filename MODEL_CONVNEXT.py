# Model and Loss functions
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Net(nn.Module):  # Net for region based segment classification
    ######################Load main net (convnext_base) class############################################
    def __init__(self):  # Load pretrained encoder and prepare net layers
        super(Net, self).__init__()

        # ---------------Load pretrained net----------------------------------------------------------
        self.Encoder1 = models.convnext_base(weights='DEFAULT')  # Load with pretrained weights

        # ----------------Change final layer to predict 512 descriptor---------------------------------
        self.Encoder1.classifier[2] = torch.nn.Linear(in_features=1024, out_features=512, bias=True)

    ###############################################Run prediction inference using the net #################
    def forward(self, Images, TrainMode=True):
        # ------------------------------- Convert from numpy to pytorch-------------------------------------------------------
        if TrainMode:
            mode = torch.float32
        else:
            mode = torch.float16

        self.type(mode)

        # Handle input images - convert from numpy to torch tensor if needed
        if isinstance(Images, np.ndarray):
            InpImages = torch.from_numpy(Images).to(device)
            InpImages = InpImages.transpose(2, 3).transpose(1, 2)
        else:
            InpImages = Images.to(device)

        InpImages = InpImages.to(mode)
        self.to(device)

        # -------------------------Normalize image---------------------------------------
        RGBMean = [123.68, 116.779, 103.939]
        RGBStd = [65, 65, 65]

        for i in range(min(3, InpImages.shape[1])):  # Handle only available channels
            InpImages[:, i, :, :] = (InpImages[:, i, :, :] - RGBMean[i]) / RGBStd[i]

        # ============================Run net layers=====================================
        # No need to concatenate a single tensor with itself
        descriptor = self.Encoder1(InpImages)  # run net
        descriptor = F.normalize(descriptor, dim=1)  # L2 normalize embeddings

        return descriptor

    ###########################################################################################
    '''cross entropy cosine loss between all descriptor in the batch'''

    def LossCosineSimilarity(self, Desc1, labels, temp=0.05):
        # Convert labels to torch tensor if needed
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(device)

        batch_size = Desc1.shape[0]

        # Create similarity matrix
        sim_matrix = torch.mm(Desc1, Desc1.T)

        # Create mask for positive pairs (same label)
        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)
        positive_mask = labels_matrix.to(device)

        # Remove self-comparisons
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & eye_mask

        # Count number of valid triplets
        num_positives = positive_mask.sum().item()
        if num_positives == 0:
            # Return zero loss if no positive pairs
            return torch.tensor(0.0, requires_grad=True, device=device), 0.0

        # Apply temperature scaling
        sim_matrix = sim_matrix / temp

        # For each anchor, compute loss over positive pairs
        correct = 0
        loss = 0
        num_triplets = 0

        for anchor_idx in range(batch_size):
            # Get positive indices for this anchor
            pos_indices = torch.where(positive_mask[anchor_idx])[0]

            if len(pos_indices) == 0:
                continue

            # For each positive pair
            for pos_idx in pos_indices:
                # All indices that are not the anchor or positive
                neg_indices = torch.where(~labels_matrix[anchor_idx])[0]

                if len(neg_indices) == 0:
                    continue

                # For this (anchor, positive) pair, compare with all negatives
                anchor_pos_sim = sim_matrix[anchor_idx, pos_idx]
                anchor_neg_sims = sim_matrix[anchor_idx, neg_indices]

                # Count correct predictions (anchor-positive > anchor-negative)
                correct += torch.sum(anchor_pos_sim > anchor_neg_sims).item()

                # For each (anchor, positive, negative) triplet
                for neg_idx in neg_indices:
                    logits = torch.stack([sim_matrix[anchor_idx, pos_idx], sim_matrix[anchor_idx, neg_idx]])
                    probs = F.softmax(logits, dim=0)
                    loss += -torch.log(probs[0] + 1e-7)  # Cross entropy focusing on positive
                    num_triplets += 1

        if num_triplets == 0:
            return torch.tensor(0.0, requires_grad=True, device=device), 0.0

        accuracy = correct / (num_positives * (batch_size - 2) + 1e-7)
        loss = loss / num_triplets

        return loss, accuracy

    ###########################################################################################
    '''cross entropy cosine loss - efficient implementation'''

    def LossCosineSimilarity_2(self, dsc, labels, temp=0.05):
        # Convert labels to torch tensor if needed
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(device)

        batch_size = dsc.shape[0]

        # Create similarity matrix
        sim_matrix = torch.mm(dsc, dsc.T)

        # Create mask for positive pairs (same label)
        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        # Create GT correlation matrix - 1 for same class (except self), 0 for different
        eye_mask = torch.eye(batch_size, device=device)
        GT_correlation = labels_matrix.float() - eye_mask

        # Zero out diagonal in similarity matrix (self comparisons)
        sim_matrix = sim_matrix * (1 - eye_mask)

        # Apply temperature scaling
        sim_matrix = sim_matrix / temp

        # Calculate softmax probabilities along each row
        exp_sim = torch.exp(sim_matrix)
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
        probs = exp_sim / exp_sim_sum

        # Get probabilities for positive pairs only
        positive_probs = probs[GT_correlation > 0]

        if positive_probs.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device), 0.0

        # Calculate loss (negative log likelihood of positive pairs)
        loss = -torch.log(positive_probs + 1e-7).mean()

        # Calculate accuracy
        predicted_indices = sim_matrix.argmax(dim=1)
        target_indices = GT_correlation.argmax(dim=1)
        accuracy = (predicted_indices == target_indices).float().mean().item()

        return loss, accuracy