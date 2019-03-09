#Generate embeddings for Categorical features

import sys
sys.path.append('../')
from modules.feature_extraction import imageFeature

imagef = imageFeature.ImageFeature("../data/AmazonHome/image_features_Home_and_Kitchen.b",
"../data/AmazonHome/imageEmbeddings.pkl",
"HOME - IMAGE EMBEDDINGS",
"../data/AmazonHome/ratings_5_core_home.csv",
True)
imagef.generateEmbeddings()
