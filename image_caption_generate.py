import os
import glob
from StyleGAN2_ada.infer_util import Infer
from Image_Captioning.utility import Catr

catr = Catr()
model_name = 'cars'
inf = Infer(network_pkl='StyleGAN2_ada/checkpoints/network-snapshot-002600.pkl',model_name = 'cars')

for i in range(0,1000000000,10):
    start = i
    end = start+10
    print('Processing ',start,' To ',end )
    inf.consecutive_inference(start, end)
    folder_path = 'StyleGAN2_ada/results/'+model_name
    catr.evaluate_folder(folder_path, model_name)
    files = glob.glob(folder_path+'/*')
    for f in files:
        os.remove(f)
