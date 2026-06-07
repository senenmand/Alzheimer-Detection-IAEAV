import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1)
        return logits / temperature

    def set_temperature(self, valid_loader, device):
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        logits_list = []
        labels_list = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, _ in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = self.model(batch_x)
                logits_list.append(logits)
                labels_list.append(batch_y)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self


def move_pipeline_to_device(model, X_train, y_train, X_test=None, y_test=None, device=None):
    """Move the model and datasets to the selected device consistently."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    X_train = X_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    if X_test is not None:
        X_test = X_test.to(device, non_blocking=True)
    if y_test is not None:
        y_test = y_test.to(device, non_blocking=True)

    return model, X_train, y_train, X_test, y_test, device


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(4, 4), padding=(1, 2)),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(0.2),
        nn.Conv2d(32, 64, kernel_size=(4, 4), padding=(1, 2)),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(0.2),
        nn.Conv2d(64, 128, kernel_size=(4, 4), padding=(1, 2)),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(0.3),
    )


def build_cnn_embedding(dropout):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.GELU(),
        nn.Dropout(dropout),
    )


def build_classifier_cnn_simple(n_classes, dropout):
    return nn.Sequential(nn.Linear(256, n_classes))


class CNN(nn.Module):
    def __init__(self, n_classes=2, dropout=0.5):
        super().__init__()
        self.features = build_cnn()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.embedding = build_cnn_embedding(dropout)
        self.classifier = build_classifier_cnn_simple(n_classes=n_classes, dropout=dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.embedding(x)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        x = self.features(x)
        x = self.pool(x)
        embedding = self.embedding(x)
        return embedding



#================== FUNCTIONS  =====================

import numpy as np
import torch
from collections import defaultdict, Counter

def compute_ece(y_true, y_prob, n_bins=10):

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        mask = (y_prob > bin_lower) & (y_prob <= bin_upper)

        if np.sum(mask) > 0:

            accuracy = np.mean(y_true[mask])

            confidence = np.mean(y_prob[mask])

            bin_prob = np.mean(mask)

            ece += np.abs(accuracy - confidence) * bin_prob

    return ece

def evaluate_by_individual(model, loader, device):

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():

        for batch_x, batch_y, ids in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            probs = F.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            prob_positive = probs[:, 1]

            all_probs.extend(prob_positive.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_ids.extend(ids)

    ind_preds = defaultdict(list)
    ind_probs = defaultdict(list)
    ind_labels = {}

    for pid, pred, prob, label in zip(
        all_ids,
        all_preds,
        all_probs,
        all_labels
    ):

        ind_preds[pid].append(pred)
        ind_probs[pid].append(prob)

        ind_labels[pid] = int(label)

    y_true = []
    y_pred = []
    y_prob = []

    for pid in ind_preds:

        final_pred = Counter(ind_preds[pid]).most_common(1)[0][0]

        final_prob = np.mean(ind_probs[pid])

        y_true.append(ind_labels[pid])
        y_pred.append(final_pred)
        y_prob.append(final_prob)

    return y_true, y_pred, y_prob

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, ys, pids = zip(*batch)
    xs = [x if x.shape[0] == 128 else x.T for x in xs]
    xs = [x.transpose(0, 1) for x in xs]   # [T,128]
    xs = pad_sequence(xs, batch_first=True)  # [B,T,128]
    xs = xs.permute(0, 2, 1)  # [B,128,T]
    xs = xs.unsqueeze(1)      # [B,1,128,T]
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys, list(pids)

def train_CNN(model,train_loader,optimizer,criterion,device):
    model.train()
    train_loss = 0

    for batch_x, batch_y, _ in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_y.size(0)

    train_loss /= len(train_loader.dataset)
    return train_loss

def val_CNN(model,val_loader,criterion,device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_x)

            loss = criterion(out, batch_y)

            val_loss += loss.item() * batch_y.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss