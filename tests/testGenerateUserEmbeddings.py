import sys
sys.path.append('../')
from modules.feature_extraction import generateUserEmbeddings

u2v = generateUserEmbeddings.User2Vec("../data/AmazonHome/ratings_5_core_home.csv", 
"../data/AmazonHome/userEmbeddings300.pkl")
u2v.generateEmbeddings()
