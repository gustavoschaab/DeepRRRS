import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration2
from modules.candidate_generation import candidateGeneration_Image
from modules.candidate_generation import featureCombination
from modules.feature_extraction import generateUserEmbeddings
from modules.feature_extraction import categoricalFeature


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181208_home.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#Generate User Embeddings with 900 dimensions
# u4096v = generateUserEmbeddings.User2Vec("../data/AmazonHome/ratings_5_core_home.csv",
# "../data/AmazonHome/userEmbeddings4096.pkl",
# 4096)
# u4096v.generateEmbeddings()

cgc = candidateGeneration_Image.CandidateGenerationImage("../data/AmazonHome/userEmbeddings4096.pkl", 
"", 
"../data/AmazonHome/imageEmbeddings.pkl",
"",
"",
"../data/AmazonHome/ratings_5_core_home.csv", 
featureCombination.EnumFeatureCombination.IMAGE,
"../data/AmazonHome", 
15,
"Home")
cgc.runNeuralNetwork()