from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.distributions import MultivariateNormal
import glob
from torchvision import transforms
from PIL import Image
from torch import randn
import open_clip
import torch
import tqdm

from torchvision.transforms.functional import pil_to_tensor, to_pil_image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.02):
        self.std = std
        self.mean = mean

        self.x = torch.linspace(0, 1, 640)
        self.y = torch.linspace(0, 1, 640)
        self.grid_x, self.grid_y = torch.meshgrid(self.x, self.y, indexing='xy')
        self.grid_points = torch.stack([self.grid_x.flatten(), self.grid_y.flatten()], dim=-1)

    def __call__(self, tensor):

        m = MultivariateNormal(torch.rand(2), torch.diag(torch.rand(2))/30)

        density_values = m.log_prob(self.grid_points)  
        density_values = torch.exp(density_values)
        density_values = density_values.view(self.grid_x.shape)

        return tensor + randn(tensor.size())*density_values * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

augmen1 = AutoAugment(policy=AutoAugmentPolicy.SVHN)
augmen2 = AddGaussianNoise()
transform=transforms.Compose([
    augmen1,
    augmen2,
])

baseline_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384', device='cuda')
baseline_model = baseline_model.visual
baseline_model.eval()
with torch.no_grad():
    res1 = {}
    res2 = {}
    res3 = {}
    for pic in tqdm.tqdm(glob.glob('/home/krzysztofkalisiak/Desktop/ROCM_repo/storage/*/*.jpg')):

        tensor_pic = pil_to_tensor(Image.open(pic))
        pa = pic.split('/storage/')[1].replace('jpg', 'pt')

        x1 = to_pil_image(transform(tensor_pic))
        x1.save(pic.replace('storage', 'storage_distorted').replace('.jpg', '1.jpg'))
        x2 = to_pil_image(transform(tensor_pic))
        x2.save(pic.replace('storage', 'storage_distorted').replace('.jpg', '2.jpg'))
        x3 = to_pil_image(transform(tensor_pic))
        x3.save(pic.replace('storage', 'storage_distorted').replace('.jpg', '3.jpg'))
        res = baseline_model(torch.stack([preprocess(x1), preprocess(x2), preprocess(x3)], axis=0).to('cuda'))
        res1[pa] = res[[0], :]
        res2[pa] = res[[1], :]
        res3[pa] = res[[2], :]

        if res3[pa].shape != (1, 1152):
            print(res3[pa].shape)
            break

torch.save(res1, '/home/krzysztofkalisiak/Desktop/ROCM_repo/embeddings/ViT-SO400M-14-SigLIP-384_mix_1/all_embeddings.pt')
torch.save(res2, '/home/krzysztofkalisiak/Desktop/ROCM_repo/embeddings/ViT-SO400M-14-SigLIP-384_mix_2/all_embeddings.pt')
torch.save(res3, '/home/krzysztofkalisiak/Desktop/ROCM_repo/embeddings/ViT-SO400M-14-SigLIP-384_mix_3/all_embeddings.pt')