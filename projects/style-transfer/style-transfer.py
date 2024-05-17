import matplotlib
import matplotlib.pyplot as plt
import PIL
import torch
from torchvision import models
from torchvision.transforms import v2

matplotlib.use("TkAgg")
torch.manual_seed(42)
torch.cuda.set_device(0) # 3090

image_size = 256
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).features.cuda()
for param in model.parameters():
    param.requires_grad = False
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

norms = {
    'squeezenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}
norm = norms['squeezenet']

transform = v2.Compose([
    v2.Resize(image_size),
    v2.ToTensor(),
    v2.Normalize(norm['mean'], norm['std']),
    v2.Lambda(lambda x: x[None])
])

transform_inv = v2.Compose([
    v2.Lambda(lambda x: x[0]),
    v2.Normalize([0, 0, 0], [1.0 / x for x in norm['std']]),
    v2.Normalize([-x for x in norm['mean']], [1, 1, 1]),
    v2.ToPILImage(),
])

def extract_features(img):
    feats = []
    prev = img
    for mod in model._modules.values():
        next = mod(prev)
        feats.append(next)
        prev = next
    return feats

def content_loss(content_weight, cur, orig):
    return content_weight * torch.sum((cur.flatten(2) - orig.flatten(2)) ** 2)

def gram(features, normalize=True):
    gram = torch.mm(features.flatten(2).squeeze(0), features.flatten(2).squeeze(0).t())
    if normalize:
        gram /= features.numel()
    return gram

def style_loss(feats, targets, layers, style_weights):
    losses = []
    for i, layer in enumerate(layers):
        diff = gram(feats[layer]) - targets[i]
        losses.append(style_weights[i] * torch.sum(diff ** 2))
    return sum(losses)

def tv_loss(img, tv_weight):
    tv_h = torch.sum((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)
    tv_w = torch.sum((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2)
    return tv_weight * (tv_h + tv_w)

def style_transfer(content_img_path, style_img_path, content_layer, content_weight, style_layers, style_weights, tv_weight):
    content_img = transform(PIL.Image.open(content_img_path)).cuda()
    content_feats = extract_features(content_img)
    content_target = content_feats[content_layer].clone()

    style_img = transform(PIL.Image.open(style_img_path)).cuda()
    style_feats = extract_features(style_img)
    style_targets = [gram(style_feats[i]) for i in style_layers]

    img = content_img.clone()
    img.requires_grad_()

    lr_init = 3.0
    lr_decayed = 0.1

    iters = 100
    decay_iter = 0.9 * iters

    optimizer = torch.optim.AdamW([img], lr=lr_init)

    for i in range(iters):
        if i < 0.95 * iters:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img)

        loss_c = content_loss(content_weight, feats[content_layer], content_target)
        loss_s = style_loss(feats, style_targets, style_layers, style_weights)
        loss_tv = tv_loss(img, tv_weight)
        loss = loss_c + loss_s + loss_tv
        loss.backward()

        if i == decay_iter:
            optimizer = torch.optim.AdamW([img], lr=lr_decayed)
        optimizer.step()

    _, ax = plt.subplots(1, 3, figsize=(24, 8), layout='tight')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[0].set_title('Content Source Img')
    ax[1].set_title('Style Source Img')
    ax[2].set_title('Transferred Img')
    ax[0].imshow(transform_inv(content_img.cpu()))
    ax[1].imshow(transform_inv(style_img.cpu()))
    ax[2].imshow(transform_inv(img.data.cpu()))
    plt.show()

dez_to_starry_night = {
    'content_img_path' : 'style-transfer/styles/dez.jpg',
    'style_img_path' : 'style-transfer/styles/starry_night.jpg',
    'content_layer' : 3,
    'content_weight' : 6e-2,
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (300_000, 1000, 15, 3),
    'tv_weight' : 5e-2
}
dez_to_the_scream = {
    'content_img_path' : 'style-transfer/styles/dez.jpg',
    'style_img_path' : 'style-transfer/styles/the_scream.jpg',
    'content_layer' : 3,
    'content_weight' : 6e-2,
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (300_000, 1000, 15, 3),
    'tv_weight' : 5e-2
}
style_transfer(**dez_to_starry_night)
style_transfer(**dez_to_the_scream)
