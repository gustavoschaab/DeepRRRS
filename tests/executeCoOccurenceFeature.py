#Generate embeddings for Categorical features

import sys
sys.path.append('../')
from modules.feature_extraction import coOccurenceFeature

cf = coOccurenceFeature.CoOccurenceFeature("../data/AmazonClothings/meta_Clothing_Shoes_and_Jewelry.json",
"../data/AmazonClothings/coOccurenceEmbeddings300.pkl",
"../data/AmazonClothings/coOccurenceEmbeddings300.model",
"../data/AmazonClothings/ratings_5_core.csv",
True)
cf.generateCoOccurenceEmbeddings()
