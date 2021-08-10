################################# Text to Image Utility ##########################################
from Image_Captioning.text_matching import TextSimilarity
from StyleGAN2_ada.infer_util import Infer
faces_ts = TextSimilarity(data_path = 'Image_Captioning/faces.csv')
faces_inf = Infer(network_pkl='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl',model_name='faces')
faces_ts = faces_ts.find_similarity('Man with smily face', threshold=0.50)
faces_inf.final_inference(seeds=faces_ts)
