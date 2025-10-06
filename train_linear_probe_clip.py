import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
import torchvision.models as models
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.fc(x)
        

args = get_args()

seed = args.seed
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 1024))).to(device)
label_log = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))).to(device)

cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 1024))).to(device)
label_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))).to(device)


train_dataset = torch.utils.data.TensorDataset(feat_log.cpu(), label_log.cpu())
val_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), label_log_val.cpu())


model = LinearProbe(1024, 1000).to(device)
breakpoint()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=15)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=15)

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

best_acc = 0
for epoch in range(30):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device).float(), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            batch_acc = outputs.argmax(1).eq(targets).float().mean().item()
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Acc: {batch_acc}")
    step_scheduler.step()
    print("LR:", step_scheduler.get_last_lr())

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device).float(), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100 * correct / total
    print(f"Epoch: {epoch}, Val Acc: {acc}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"./linear_probe.pth")
        print(f"Best Val Acc: {best_acc}")
    print("")