import sys
sys.path.append('../')
from modules.feature_extraction import textualFeature

tf = textualFeature.TextualFeature("../data/AmazonClothings/meta_Clothing_Shoes_and_Jewelry.json", "../data/AmazonClothings/textual_embeddings.pkl", "../data/AmazonClothings/textual_embeddings.model", 10, 300, 0.025)
t = tf.generateEmbeddings()
