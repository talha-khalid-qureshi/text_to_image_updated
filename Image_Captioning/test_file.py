import os
from utility import Catr
catr = Catr()

input_folder_path = '../StyleGAN2-ada/results/faces'
for im in os.listdir(input_folder_path):
    if im.endswith(('.png', '.jpeg', '.jpg')):
        print('Processing : ', im)
        results = catr.evaluate(os.path.join(input_folder_path, im),input_folder_path.split('/')[-1])
        print(results)