# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional
import click
from StyleGAN2_ada import dnnlib
import numpy as np
import PIL.Image
import torch
from StyleGAN2_ada import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    d = s
    s = ''
    s = ', '.join([str(item) for item in d])
    range_re = re.compile(r'^(\d+)-(\d+)$')
    # print('--------- ------------s :',s)
    # print('--------- ------------range_re :',range_re)

    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

class Infer():
    def __init__(self,network_pkl,model_name):
        self.truncation_psi = 1
        self.noise_mode = 'const'
        self.outdir = 'StyleGAN2_ada/results/'+model_name
        self.projected_w = None
        print('Loading networks from "%s"...' % network_pkl)
        self.device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        os.makedirs(self.outdir, exist_ok=True)


    def final_inference(self,seeds):
        # print('<-----------------',seeds)
        seeds = num_range(seeds)
        # print('------------>',seeds)
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            img = self.G(z, self.label, truncation_psi=self.truncation_psi, noise_mode=self.noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{self.outdir}/seed{seed:04d}.png')
            
    def final_generation(self, seeds):
        # print('<-----------------',seeds)
        seeds = num_range(seeds)
        # print('------------>',seeds)
        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            img = self.G(z, self.label, truncation_psi=self.truncation_psi, noise_mode=self.noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            file_path = f'{self.outdir}/seed{seed:04d}.png'
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(file_path)
            file_name = f'seed{seed:04d}.png'
#             s3_path = f"faces-models/{input_text_data.replace(' ', '_')}/{file_name}"
#             with open (file_path, 'rb') as data:
#                 self.bucket.put_object(Key=s3_path, Body=data) 
            #os.remove(file_path)

    
    def consecutive_inference(self, start, end):
        seeds = [i for i in range(start, end)]
        print('---------------->>>>>>>>>>>',seeds)
        self.final_generation(seeds)
        
#----------------------------------------------------------------------------
#"""
if __name__ == "__main__":
    inf = Infer(network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl',model_name='faces')
    inf.final_inference(seeds=[5,35,6,25])
    inf.consecutive_inference(start=10, end=20)
#"""
#----------------------------------------------------------------------------
