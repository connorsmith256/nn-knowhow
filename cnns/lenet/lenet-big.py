import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

matplotlib.use("TkAgg")

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0, ), (1.0, ))
])

imagenet_full = datasets.MNIST(root="cnns/lenet/datasets", transform=transform, train=True, download=True)
n_train = int(0.9 * len(imagenet_full))
n_val = len(imagenet_full) - n_train
imagenet_tr, imagenet_val = random_split(imagenet_full, [n_train, n_val])
imagenet_te = datasets.MNIST(root="cnns/lenet/datasets", transform=transform, train=False)

batch_size_tr = 200
batch_size_val = 20
batch_size_te = 1000

dataloader_tr = DataLoader(imagenet_tr, batch_size=batch_size_tr, shuffle=True)
dataloader_val = DataLoader(imagenet_val, batch_size=batch_size_val, shuffle=True)
dataloader_te = DataLoader(imagenet_te, batch_size=batch_size_te, shuffle=True)

print(f'{len(imagenet_tr)} train images ({len(dataloader_tr)} batches)')
print(f'{len(imagenet_val)} train images ({len(dataloader_val)} batches)')
print(f'{len(imagenet_te)} train images ({len(dataloader_te)} batches)')

# init model
torch.manual_seed(42)
# g = torch.Generator().manual_seed(2147483647)
# g = torch.Generator(device='cuda').manual_seed(2147483647)
sample_g = torch.Generator(device='cuda').manual_seed(2147483647 + 10)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        with torch.device('cuda'):
            # 5x5 conv, padding 2
            self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, padding=2)
            self.c2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=2)
            # 2x2 avg pool, stride 2
            self.p1 = nn.MaxPool2d(kernel_size=3, stride=2)
            # 5x5 conv, padding 0
            self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=0)
            self.c4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
            # 2x2 avg pool, stride 2
            self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)

            self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=0)
            self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=0)
            self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, padding=0)

            self.p3 = nn.MaxPool2d(kernel_size=1, stride=2)

            # linear 120
            self.l1 = nn.Linear(in_features=256, out_features=4096)
            # linear 84
            self.l2 = nn.Linear(in_features=4096, out_features=1000)
            # linear 10
            self.l3 = nn.Linear(in_features=1000, out_features=10)

            # self.net = nn.Sequential(
            #     c1,
            #     p1,
            #     c2,
            #     p2,
            #     l1,
            #     l2,
            #     l3
            # )

    def forward(self, x, batch_size):
        x = self.c1(x)
        x = self.c2(x)
        x = self.p1(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.p2(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.p3(x)
        x = self.l1(x.view(batch_size, -1))
        x = self.l2(x)
        x = self.l3(x)
        return x
        # return self.net(x)

model = LeNet()
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# test
@torch.no_grad()
def test_accuracy():
    correct_samples = []
    incorrect_samples = []

    for data, targets in dataloader_te:
        data, targets = data.cuda(), targets.cuda()

        logits = model(data, batch_size_te)
        _, predicted = torch.max(logits.data, 1)
        mask_correct = (predicted == targets)
        mask_incorrect = ~mask_correct

        correct_data = data[mask_correct]
        correct_predicted = predicted[mask_correct]
        correct_true = targets[mask_correct]
        for i in range(correct_data.shape[0]):
            correct_samples.append((correct_data[i], correct_predicted[i], correct_true[i]))

        incorrect_data = data[mask_incorrect]
        incorrect_predicted = predicted[mask_incorrect]
        incorrect_true = targets[mask_incorrect]
        for i in range(incorrect_data.shape[0]):
            incorrect_samples.append((incorrect_data[i], incorrect_predicted[i], incorrect_true[i]))

    return correct_samples, incorrect_samples

# train
num_epochs = 5
train_losses = []
val_losses = []
ud = []

for i in range(num_epochs):
    model.train()
    train_losses.append(0)
    for j, (data, targets) in enumerate(dataloader_tr):
        optimizer.zero_grad()

        data, targets = data.cuda(), targets.cuda()

        logits = model(data, batch_size_tr)
        B, C = logits.shape
        loss = criterion(logits, targets)
        train_losses[-1] += loss.item() / len(dataloader_tr)

        loss.backward()
        optimizer.step()

    model.eval()
    val_losses.append(0)
    with torch.no_grad():
        for j, (data, targets) in enumerate(dataloader_val):
            data, targets = data.cuda(), targets.cuda()

            logits = model(data, batch_size_val)
            B, C = logits.shape
            loss = criterion(logits, targets)

            val_losses[-1] += loss.item() / len(dataloader_val)

    if False:
        correct, incorrect = test_accuracy()
        accuracy = 100.0 * len(correct) / (len(correct) + len(incorrect))
        print(f'{i+1:3d}/{num_epochs} Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}, Test: {accuracy:.4f}%')
    else:
        print(f'{i+1:3d}/{num_epochs} Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}')

# visualize loss
plt.plot(np.linspace(0, 1, len(train_losses)), train_losses)
plt.plot(np.linspace(0, 1, len(val_losses)), val_losses)
plt.legend(['train', 'val'])
plt.show()

# test
def plot_samples(samples, class_names, title, num_samples=10):
    num_col = 10
    num_row = num_samples // num_col + 1
    figsize = (num_col * 1, num_row * 1) # 1 inch square

    plt.figure(figsize=figsize)
    for i, (img, pred, true) in enumerate(samples[:num_samples]):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(num_row, num_col, i + 1)
        plt.imshow(img)
        plt.title(f'P: {class_names[pred]}, T: {class_names[true]}')
        plt.axis('off')
    plt.tight_layout(pad=0.0, rect=[0, 0, 1, 0.95])
    plt.suptitle(title, fontsize=10, y=0.98)
    plt.show()

model.eval()
correct, incorrect = test_accuracy()
accuracy = 100.0 * len(correct) / (len(correct) + len(incorrect))
print(f'Final accuracy: {accuracy:.4f}%')
# plot_samples(correct, [i for i in range(10)], "Correct", 100)
plot_samples(incorrect, [i for i in range(10)], "Incorrect", 100)

# visualize layers
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(model.layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
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
