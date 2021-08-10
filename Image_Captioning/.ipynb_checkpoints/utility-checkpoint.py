import torch
import pandas as pd

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

parser = argparse.ArgumentParser(description='Image Captioning')
#parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)

class Catr():
    def __init__(self):
        
        args = parser.parse_args()
        #image_path = args.path
        version = args.v
        self.checkpoint_path = args.checkpoint
        self.config = Config()

        if version == 'v1':
            self.model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
        elif version == 'v2':
            self.model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
        elif version == 'v3':
            self.model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        else:
            print("Checking for checkpoint.")
            if checkpoint_path is None:
                raise NotImplementedError('No model to chose from!')
            else:
                if not os.path.exists(checkpoint_path):
                    raise NotImplementedError('Give valid checkpoint path')
                print("Found checkpoint! Loading!")
                self.model,_ = caption.build_model(self.config)
                print("Loading Checkpoint...")
                checkpoint = torch.load(self.checkpoint_path, map_location='cuda')
                self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)
            
        

        def create_caption_and_mask(self,start_token, max_length):
            caption_template = torch.zeros((1, max_length), dtype=torch.long)
            mask_template = torch.ones((1, max_length), dtype=torch.bool)

            caption_template[:, 0] = start_token
            mask_template[:, 0] = False

            return caption_template, mask_template


        self.caption, self.cap_mask = create_caption_and_mask(self,
            self.start_token, self.config.max_position_embeddings)


    @torch.no_grad()
    def evaluate(self,image_path, model_name):
        if os.path.exists(model_name+'.csv'):
            df = pd.read_csv(model_name+'.csv')
            df = df.drop(columns=['Unnamed: 0'])
        else:
            df = pd.DataFrame(columns=['Captions', 'Seed_Value'])
        seed_value = os.path.basename(image_path)
        # print('Based Name : ', seed_value)
        seed_value = seed_value.split('.')[0]
        # print('Seed Value : ', seed_value)
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        for i in range(self.config.max_position_embeddings - 1):
            predictions = self.model(image, self.caption, self.cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                # return caption
                final_caption = self.tokenizer.decode(self.caption[0].tolist(), skip_special_tokens=True).capitalize()
                df.loc[len(df)] = [final_caption,seed_value[4:]]
                df.to_csv(os.path.join(os.getcwd(),model_name+'.csv'))
                return final_caption
            self.caption[:, i+1] = predicted_id[0]
            self.cap_mask[:, i+1] = False
        final_caption = self.tokenizer.decode(self.caption[0].tolist(), skip_special_tokens=True).capitalize()

        df.loc[len(df)] = [final_caption,seed_value[4:]]
        df.to_csv(os.path.join(os.getcwd(),model_name+'.csv'))

        return final_caption
"""
if __name__ == '__main__':
    catr = Catr()
    output = catr.evaluate('29.jpg','car')
    print(output)
"""