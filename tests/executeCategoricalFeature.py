#Generate embeddings for Categorical features

import sys
sys.path.append('../')
from modules.feature_extraction import categoricalFeature

cf = categoricalFeature.CategoricalFeature("../data/AmazonHome/meta_home.json",
"../data/AmazonHome/glove.6B.300d.txt",
"../data/AmazonHome/categoricalEmbeddings300.pkl",
"../data/AmazonHome/ratings_5_core_home.csv",
True)
cf.generateCategoricalEmbeddings()