import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
from datasets import load_dataset
from warmup_scheduler import GradualWarmupScheduler

matplotlib.use("TkAgg")
torch.manual_seed(42)
torch.cuda.set_device(0) # 3090

batch_size_tr = 1024
batch_size_val = 1024
batch_size_te = 1024

# lr
base_lr = 2.5e-4
start_lr = base_lr * (batch_size_tr / 128) # linear scaling rule. OG paper had 1e-2 for 128
lr_weight_decay = 1e-2
lr_momentum = 0.9
lr_decay_period = 50
lr_decay_gamma = 0.1 # decay 10x
lr_warmup_multiplier = 1.0 # start at 0, end at lr
lr_warmup_epochs = 5
# epochs
num_epochs = 1000
max_patience = 50
# subsets
sample_subset = 0.5
full_every = 10

norms = {
    'imagenette_full': {
        # 'mean': [0.4661, 0.4581, 0.4292],
        # 'std': [0.2393, 0.2327, 0.2405]
    },
    'cifar10_32': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    }
}
norms = norms['cifar10_32']

image_size = 32
transform_tr = v2.Compose([
    # v2.Resize(image_size),
    # v2.CenterCrop(image_size),
    v2.Pad(4),
    # v2.RandomRotation(30),
    v2.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
    # v2.TrivialAugmentWide(),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(image_size),
    # v2.RandomResizedCrop(image_size, scale=(1, 1), ratio=(1, 1)),
    v2.ToTensor(),
    v2.Normalize(norms['mean'], norms['std']),
])

transform_te = transforms.v2.Compose([
    # transforms.v2.Resize(image_size),
    # transforms.v2.CenterCrop(image_size),
    transforms.v2.ToTensor(),
    transforms.v2.Normalize(norms['mean'], norms['std']),
])

# source = datasets.Imagenette
source = datasets.CIFAR10
imagenet_tr = source(root="cnns/imagenet/datasets", transform=transform_tr, train=True, download=True)
# imagenet_tr = source(root="cnns/imagenet/datasets", transform=transform_tr, split='train', size="full")
# imagenet_tr, _ = random_split(imagenet_tr, [0.025, 0.975]) # TODO REMOVE

imagenet_val_full = source(root="cnns/imagenet/datasets", transform=transform_te, train=False)
# imagenet_val_full = source(root="cnns/imagenet/datasets", transform=transform_te, split='val', size="full")
n_val = int(0.5 * len(imagenet_val_full))
n_test = len(imagenet_val_full) - n_val
imagenet_val, imagenet_te = random_split(imagenet_val_full, [n_val, n_test])
# imagenet_val, _ = random_split(imagenet_val, [0.1, 0.9]) # TODO REMOVE

dataloader_tr = DataLoader(imagenet_tr, batch_size=batch_size_tr)
dataloader_val = DataLoader(imagenet_val, batch_size=batch_size_val)
dataloader_te = DataLoader(imagenet_te, batch_size=batch_size_te)

print(f'{len(imagenet_tr)} train images ({len(dataloader_tr)} batches of {batch_size_tr})')
print(f'{len(imagenet_val)} val images ({len(dataloader_val)} batches of {batch_size_val})')
print(f'{len(imagenet_te)} test images ({len(dataloader_te)} batches of {batch_size_te})')

# gather stats
def get_stats():
    imagenet_tr_stat = source(root="cnns/imagenet/datasets", transform=transforms.Compose([transforms.ToTensor()]), train=True, download=True)
    stats_loader = DataLoader(imagenet_tr_stat, batch_size=len(imagenet_tr_stat))
    images, _ = next(iter(stats_loader))
    images = images.view(images.size(0), images.size(1), -1)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    mean = images.mean([0, 2])
    std = images.std([0, 2])
    print(f'mean: {mean}, std: {std}')
if False:
    get_stats()

class ResBlock(nn.Module):
    def __init__(self, out_channels, down_sample=False):
        super(ResBlock, self).__init__()

        self.down_sample = down_sample
        in_channels = out_channels // 2 if down_sample else out_channels

        self.relu = nn.ReLU()
        # 3x3
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=2 if down_sample else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # identity
        self.ci = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, padding=0, stride=2 if down_sample else 1)
        self.bni = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.c1(x)))
        c2 = self.bn2(self.c2(c1))

        ci = self.bni(self.ci(x))

        sum = c2 + (ci if self.down_sample else x)
        out = self.relu(sum)
        return out

class ResBlockBottleneck(nn.Module):
    def __init__(self, out_channels, down_sample=False):
        super(ResBlockBottleneck, self).__init__()

        self.down_sample = down_sample
        in_channels = out_channels // 2 if down_sample else out_channels

        self.relu = nn.ReLU()
        # 1x1
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2 if down_sample else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1
        self.c3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # identity
        self.ci = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2 if down_sample else 1)
        self.bni = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.c1(x)))
        c2 = self.relu(self.bn2(self.c2(c1)))
        c3 = self.bn3(self.c3(c2))

        ci = self.bni(self.ci(x))

        sum = c3 + (ci if self.down_sample else x)
        out = self.relu(sum)
        return out

class ResLayer(nn.Module):
    def __init__(self, channels, block_type, num_blocks, down_sample=True):
        super(ResLayer, self).__init__()

        self.blocks = nn.ModuleList([block_type(
            channels,
            down_sample=i == 0 and down_sample
        ) for i in range(num_blocks)])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        base_channels = 64

        self.c0 = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=7, padding=3, stride=2)
        self.bn0 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # ResNet 34
        block_type = ResBlockBottleneck
        block_depths = [3, 4, 6, 3]
        # block_type = ResBlock
        # block_depths = [3, 3, 3] # ResNet 18, CIFAR10

        self.layers = nn.ModuleList([ResLayer(
            base_channels * pow(2, i),
            block_type,
            num_blocks=num_blocks,
            down_sample=i > 0
        ) for i, num_blocks in enumerate(block_depths)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=base_channels * pow(2, len(self.layers) - 1), out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, ResBlockBottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        maxpool = self.maxpool(self.relu(self.bn0(self.c0(x))))

        conv = maxpool
        for i in range(len(self.layers)):
            conv = self.layers[i](conv)

        avgpool = self.avgpool(conv)
        logits = self.fc1(torch.flatten(avgpool, 1))
        return logits

# model = ResNet().cuda()
model = models.resnet34().cuda()

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=start_lr, weight_decay=lr_weight_decay)

warmup_lr = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (start_epoch + epoch + 1) / lr_warmup_epochs)
main_lr = lr_scheduler.ReduceLROnPlateau(optimizer, patience=30) # defaults to factor=0.1, patience=10
# scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, main_lr], milestones=[lr_warmup_epochs])

# load
try:
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # warmup_lr.load_state_dict(checkpoint['warmup_lr_state_dict'])
    main_lr.load_state_dict(checkpoint['main_lr_state_dict'])
    # start_epoch = checkpoint['epoch']
    start_epoch = 124
    
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    val_loss = checkpoint['val_loss']
    val_acc = checkpoint['val_acc']
    test_loss = checkpoint['test_loss']
    test_acc = checkpoint['test_acc']
    
    best_loss = val_loss[-1]
    print(f'loaded from epoch {start_epoch}, loss {best_loss:.4f}')
except Exception as e:
    print(e)
    start_epoch = 0

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []

    best_loss = float('inf')
    print('started from scratch')

model = model.cuda()

if start_epoch < lr_warmup_epochs:
    lr = warmup_lr.get_last_lr()[0]
else:
    lr = main_lr.get_last_lr()[0]

def subset(dataloader):
    total_size = len(dataloader.dataset)
    indices = torch.randperm(total_size)[:int(sample_subset * total_size)]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    return DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=sampler, drop_last=dataloader.drop_last)

# test
@torch.no_grad()
def check_accuracy(dataloader, return_samples=False):
    correct_samples = []
    incorrect_samples = []

    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()

        logits = model(data)
        _, predicted = torch.max(logits.data, 1)
        mask_correct = (predicted == targets)
        mask_incorrect = ~mask_correct

        correct_data = data[mask_correct].cpu()
        correct_predicted = predicted[mask_correct].cpu()
        correct_true = targets[mask_correct].cpu()
        for i in range(correct_data.shape[0]):
            if return_samples:
                correct_samples.append((correct_data[i], correct_predicted[i], correct_true[i]))
            else:
                correct_samples.append(1)

        incorrect_data = data[mask_incorrect].cpu()
        incorrect_predicted = predicted[mask_incorrect].cpu()
        incorrect_true = targets[mask_incorrect].cpu()
        for i in range(incorrect_data.shape[0]):
            if return_samples:
                incorrect_samples.append((incorrect_data[i], incorrect_predicted[i], incorrect_true[i]))
            else:
                incorrect_samples.append(1)

    accuracy = 100.0 * len(correct_samples) / (len(correct_samples) + len(incorrect_samples))
    return accuracy, correct_samples, incorrect_samples

# train
patience = max_patience
ud = []
ts = time.time()
try:
    for i in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        t0 = time.time()
        
        model.train()
        train_loss.append(0)
        for j, (data, targets) in enumerate(dataloader_tr):
            data, targets = data.cuda(), targets.cuda()
            logits = model(data)
            loss = criterion(logits, targets)
            train_loss[-1] += loss.item() / len(dataloader_tr)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        do_sample = i % full_every != 0
        with torch.no_grad():
            t1 = time.time()
            data_tr = subset(dataloader_tr) if do_sample else dataloader_tr
            train_acc.append(check_accuracy(data_tr)[0])
            t2 = time.time()
        
            data_val = dataloader_val #subset(dataloader_val) if do_sample else dataloader_val
            val_loss.append(0)
            for j, (data, targets) in enumerate(data_val):
                data, targets = data.cuda(), targets.cuda()
                logits = model(data)
                loss = criterion(logits, targets)
                val_loss[-1] += loss.item() / len(data_val)
            t3 = time.time()
            val_acc.append(check_accuracy(data_val)[0])
            t4 = time.time()

            data_te = dataloader_te #subset(dataloader_te) if do_sample else dataloader_te
            test_loss.append(0)
            for j, (data, targets) in enumerate(data_te):
                data, targets = data.cuda(), targets.cuda()
                logits = model(data)
                loss = criterion(logits, targets)
                test_loss[-1] += loss.item() / len(data_te)
            t5 = time.time()
            test_acc.append(check_accuracy(data_te)[0])
            t6 = time.time()

        if i < lr_warmup_epochs:
            lr = warmup_lr.get_last_lr()[0]
            warmup_lr.step()
        else:
            lr = main_lr.get_last_lr()[0]
            main_lr.step(metrics=val_loss[-1])

        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            patience = max_patience
            torch.save({
                'epoch': i,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_lr_state_dict': warmup_lr.state_dict(),
                'main_lr_state_dict': main_lr.state_dict(),
            }, 'cnns/imagenet/models/best_model.pth')
            saved_str = 'S'
        else:
            saved_str = ' '
            patience -= 1
            if patience == 0:
                print('Stopping, out of patience')
                break

        print(f'[{" " if do_sample else "F"}] {i:3d}/{num_epochs}', end=', ')
        print(f'Tr: {train_loss[-1]:6.4f} ({(t1 - t0):4.1f}s)', end = ', ')
        print(f'{train_acc[-1]:6.2f}% ({(t2 - t1):4.1f}s)', end=', ')
        print(f'[{saved_str}]', end= ' ')
        print(f'Val: {val_loss[-1]:6.4f} ({(t3 - t2):4.1f}s)', end=', ')
        print(f'{val_acc[-1]:6.2f}% ({(t4 - t3):4.1f}s)', end=', ')
        print(f'Test: {test_loss[-1]:6.4f} ({(t5 - t4):4.1f}s)', end=', ')
        print(f'{test_acc[-1]:6.2f}% ({(t6 - t5):4.1f}s)', end=', ')
        print(f'Ti: {(t6 - t0):5.1f}s, Tt: {(time.time() - ts):6.1f}s, LR: {lr:.4f}')

except KeyboardInterrupt:
    print('Stopping, user requested')
tf = time.time()
print(f'Trained for {(tf-ts):4.1f}s')

# visualize loss
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train', 'val'])
plt.show()

# final test
def plot_samples(samples, class_names, title, num_samples=10):
    num_col = 10
    num_row = num_samples // num_col + 1
    figsize = (num_col * 1, num_row * 1) # 1 inch square

    plt.figure(figsize=figsize)
    for i, (img, pred, true) in enumerate(samples[:num_samples]):
        img = img.cpu()
        
        # denorm
        for t, mean, std in zip(img, norms['mean'], norms['std']):
            t.mul_(std).add_(mean)

        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(num_row, num_col, i + 1)
        plt.imshow(img)
        plt.title(f'P: {class_names[pred]}, T: {class_names[true]}')
        plt.axis('off')
    plt.tight_layout(pad=0.0, rect=[0, 0, 1, 0.95])
    plt.suptitle(title, fontsize=10, y=0.98)
    plt.show()

model.eval()
acc, correct, incorrect = check_accuracy(dataloader_te, return_samples=True)
print(f'Final accuracy: {acc:6.2f}%')
# plot_samples(correct, [i for i in range(10)], "Correct", 100)
# plot_samples(incorrect, [i for i in range(10)], "Incorrect", 100)

# # visualize layers
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(model.layers[:-1]): # omit last layer
#     if isinstance(layer, nn.ReLU):
#         t = layer.out
#         print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('activation distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(model.layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
#         t = layer.out.grad
#         print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('gradient distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#     t = p.grad.cpu()
#     if p.ndim == 2:
#         print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'{i} {tuple(p.shape)}')
# plt.legend(legends);
# plt.title('weights gradient distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#   if p.ndim == 2:
#     plt.plot([ud[j][i] for j in range(len(ud))])
#     legends.append('param %d' % i)
# plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
# plt.legend(legends);
# plt.title('log ratio of gradient to data')
# plt.show()
