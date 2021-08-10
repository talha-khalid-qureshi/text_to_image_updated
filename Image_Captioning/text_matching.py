from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np


class TextSimilarity:
  def __init__(self,data_path):
    self.model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
    df = pd.read_csv(data_path)
    self.captions = df['Captions']
    self.seed_value = df['Seed_Value']
    self.embeddings1 = self.model.encode(self.captions, convert_to_tensor=True).cpu()
  
  # def make_embeddings(self,data_path):
  #   df = pd.read_csv(data_path)
  #   captions = df['Captions']
  #   embeddings1 = self.model.encode(captions, convert_to_tensor=True)
  #   return embeddings1

  def find_similarity(self, text,threshold=0.40):
    embeddings2 = self.model.encode([text], convert_to_tensor = True).cpu()
    cosine_scores = util.pytorch_cos_sim(self.embeddings1, embeddings2)
    # For All Values
    # sorted_index = tf.argsort(cosine_scores.flatten().tolist()).numpy()[::-1]
    sorted_index = np.where(cosine_scores> threshold)[0]

#     return self.captions[sorted_index].tolist(), self.seed_value[sorted_index].tolist()
    return self.seed_value[sorted_index].tolist()

if __name__ == '__main__':
  ts = TextSimilarity(data_path = 'Image_Captioning/faces.csv')
  print(ts.find_similarity('Man with smily face', threshold=0.40))