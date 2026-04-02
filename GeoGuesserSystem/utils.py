import numba as nb
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn

def batchify_tensor(sample, device=None, dtype=None):
    if isinstance(sample, (int, float, np.number)):
        return torch.tensor([sample], device=device, dtype=dtype)
    elif isinstance(sample, np.ndarray):
        return torch.tensor(sample, device=device, dtype=dtype).unsqueeze(0)
    elif isinstance(sample, torch.Tensor):
        return sample.unsqueeze(0).to(device=device, dtype=dtype) if (device or dtype) else sample.unsqueeze(0)
    elif isinstance(sample, str):
        return [sample]
    elif isinstance(sample, (list, tuple)):
        return type(sample)(batchify_tensor(x, device=device, dtype=dtype) for x in sample)
    else:
        return [sample]

@nb.njit()
def haversine_distance(p1x,p1y,p2x,p2y):
    r = 6371.001
    p = np.radians(np.array([p1x, p1y, p2x, p2y]))
    return 2*r*np.arcsin(np.sqrt(
        (np.sin((p[1]-p[3])/2)**2)+(np.cos(p[1])*np.cos(p[3])*np.sin((p[0]-p[2])/2)**2)
        ))
@nb.njit()
def f(pct_n, shp_n):

    full_precompute = np.empty((pct_n.shape[0], shp_n.shape[0]), dtype=np.float32)

    for x in np.arange(pct_n.shape[0]):
        for y in np.arange(shp_n.shape[0]):
            full_precompute[x, y] = haversine_distance(
            pct_n[x, 0], pct_n[x, 1], shp_n[y, 0], shp_n[y, 1])
    return full_precompute

def HaversineLoss(output, y):

    loss = torch.mean(-torch.sum(torch.log(output) * y, 1))

    return loss

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam


#@title Helper functions

#@markdown Some helper functions for overlaying heatmaps on top
#@markdown of images and visualizing with matplotlib.

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    return axes
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

def FineTuneGlobalIterator(loss_multiplier, tau, layer1, layer2, drp):

        system_conf = system_configs['SYS_ID_grid0']
        system_conf["loss_multiplier"][2][0] = loss_multiplier

        system_conf['tau'] = tau

        layer1 = int(layer1)
        layer2 = int(layer2)

        extension = [
                [
                    [nn.Dropout(p=drp), nn.Linear(1152, layer1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)]
                ],
                [ 
                    [nn.ReLU(), nn.Dropout(p=drp), nn.Linear(layer1+17, layer2)]
                ],
                [ 
                    [nn.ReLU(), nn.Dropout(p=drp), nn.Linear(layer2, 1410)]
                ],
                [
                    [nn.Softmax(dim=1)]
                ]
            ]

        model = GeoBrainNetwork(None, None, extension,
                                {
                                    0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                                    1:[False],
                                    2:[True],
                                    3:[True]
                                },
                                {
                                    0:torch.cat,
                                    1:first,
                                    2:first,
                                    3:first
                                },
                                {
                                    0:[('side_tasks', slice(0, 16))],
                                    2:[('geolocation', slice(0, 1))]
                                }).to('cuda')
        
        optimizer = optim.AdamW(model.parameters(), **{'lr':0.0001})

        id = np.random.choice(np.arange(1000))

        if os.path.isfile('hypertunings/config_%i.json'%id):
            id = np.random.choice(np.arange(1000))

        BR = BRAIN()
        BR.NN = model.to(DEVICE)
        BR.train_dataset = train_dataset
        BR.test_dataset = test_dataset
        BR.loss = system_conf['auxiliary_loss']
        BR.loss_multiplier = system_conf['loss_multiplier']
        BR.tau = system_conf['tau']
        BR.pct_n = pct_n
        BR.pct = pct
        BR.mapping = mapping
        BR.shp_n = shp_n
        BR.shp = shp
        BR.device = DEVICE
        BR.optimizer = optimizer
        BR.y_variable_names = system_conf['variable_names']
        BR.batch_size=system_conf['batch_size']

        BR.prepare_system(list(countries))

        BR.prepare_dataloaders()
        BR.train(40, name='hypertunings/Version %i' % id)
        BR.generate_test_main(on='test')
        
        acc = {}
        for j in [2, 4, 6, 12]:
            t = BR.task_summary_test['geolocation'].map(lambda x: x[:j]).groupby(['pred'])['real'].value_counts().unstack(-1)
            acc[j] = np.nansum(np.diag(t))/np.nansum(t)

        y = BR.task_summary_test.drop(columns='geolocation')
        SCORE = 0
        for c in y.columns.get_level_values(0).unique():
            SCORE-=(y[c].diff(axis=1)['real']**2).mean()
        SCORE += acc[4]*20

        params = {
            'loss_multiplier':float(loss_multiplier), 
            'tau':float(tau), 
            'layer1':int(layer1), 
            'layer2':int(layer2), 
            'drp':float(drp),
            'SCORE':float(SCORE),
            'id':int(id)
        }

        with open('hypertunings/config_%i.json'%id, 'w', encoding='utf-8') as f:
            json.dump(params, f)

        BR.task_summary_test.to_csv('hypertunings/df_%i.csv' % id)

        del BR
        gc.collect()
        torch.cuda.empty_cache()

        return SCORE