import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration2
from modules.candidate_generation import candidateGeneration_categorical
from modules.candidate_generation import candidateGeneration_cooccurence
from modules.candidate_generation import candidateGeneration_TextualCategorical
from modules.candidate_generation import candidateGeneration_TextualCategoricalCoOccurence
from modules.candidate_generation import candidateGeneration_CategoricalCoOccurence
from modules.candidate_generation import candidateGeneration_TextualCoOccurence
from modules.candidate_generation import featureCombination
from modules.feature_extraction import generateUserEmbeddings
from modules.feature_extraction import textualFeature
from modules.feature_extraction import categoricalFeature
from modules.feature_extraction import coOccurenceFeature
#from utils import transform_ratings_file

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181129_clothings.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

directory = "AmazonClothings"
csfFile = "ratings_5_core.csv"
metaFile = "meta_Clothing_Shoes_and_Jewelry.json"
datasetName = "Clothings"

# tf = transform_ratings_file.TranformFile("../data/AmazonClothings/reviews_Clothing_Shoes_and_Jewelry_5.json", "../data/AmazonClothings/ratings_5_core.csv")
# tf.transformFile()

#Generate User Embeddings with 300 dimensions
# u300v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile, 
# "../data/" + directory + "/userEmbeddings300.pkl",
# 300)
# u300v.generateEmbeddings()

# #Generate User Embeddings with 600 dimensions
# u600v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
# "../data/" + directory + "/userEmbeddings600.pkl",
# 600)
# u600v.generateEmbeddings()

# #Generate User Embeddings with 900 dimensions
# u900v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
# "../data/" + directory + "/userEmbeddings900.pkl",
# 900)
# u900v.generateEmbeddings()

#Extract textual features and generate embeddings with 300 dimensions
# tf = textualFeature.TextualFeature("../data/" + directory + "/" + metaFile, 
# "../data/" + directory + "/textual_embeddings.pkl", 
# "../data/" + directory + "/textual_embeddings.model", 10, 300, 0.025)
# t = tf.generateEmbeddings()

# #Extract categorical features and generate embeddings with 300 dimensions
# cf = categoricalFeature.CategoricalFeature("../data/" + directory + "/" + metaFile,
# "../data/" + directory + "/glove.6B.300d.txt",
# "../data/" + directory + "/categoricalEmbeddings300.pkl",
# "../data/" + directory + "/ratings_5_core_home.csv",
# False)
# cf.generateCategoricalEmbeddings()

#Extract co-occurence features and generate embeddings with 300 dimensions
# ccf = coOccurenceFeature.CoOccurenceFeature("../data/" + directory + "/" + metaFile,
# "../data/" + directory + "/coOccurenceEmbeddings300.pkl",
# "../data/" + directory + "/coOccurenceEmbeddings300.model",
# "../data/" + directory + "/" + csfFile,
# True)
# ccf.generateCoOccurenceEmbeddings()

#Neural Network Textual
cg = candidateGeneration2.CandidateGeneration("../data/" + directory + "/userEmbeddings300.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"",
"",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL,
"../data/" + directory, 
15,
datasetName)
cg.runNeuralNetwork()

#Neural Network Categorical
cgc = candidateGeneration_categorical.CandidateGenerationCategorical("../data/" + directory + "/userEmbeddings300.pkl", 
"", 
"",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.CATEGORICAL,
"../data/" + directory, 
15,
datasetName)
cgc.runNeuralNetwork()

#Neural Network Co-Occurence
cgcO = candidateGeneration_cooccurence.CandidateGenerationCooccurence("../data/" + directory + "/userEmbeddings300.pkl", 
"", 
"",
"",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.COOCCURENCE,
"../data/" + directory, 
15,
datasetName)
cgcO.runNeuralNetwork()

#Neural Network Textual + Categorical
cgtc = candidateGeneration_TextualCategorical.CandidateGenerationTextualCategorical("../data/" + directory + "/userEmbeddings600.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_CATEGORICAL,
"../data/" + directory, 
15,
datasetName)
cgtc.runNeuralNetwork()

#Neural Network Textual + Categorical + Co-Occurence
cgtcc = candidateGeneration_TextualCategoricalCoOccurence.CandidateGenerationTextualCategoricalCoOccurence("../data/" + directory + "/userEmbeddings900.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_CATEGORICAL_COOCCURENCE,
"../data/" + directory, 
15,
datasetName)
cgtcc.runNeuralNetwork()

#Neural Network Categorical + Co-Occurence
cgtCatCo = candidateGeneration_CategoricalCoOccurence.CandidateGenerationCategoricalCoOccurence("../data/" + directory + "/userEmbeddings600.pkl", 
"", 
"",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.CATEGORICAL_COOCCURENCE,
"../data/" + directory, 
15,
datasetName)
cgtCatCo.runNeuralNetwork()

#Neural Network Textual + Co-Occurence
cgtTexCo = candidateGeneration_TextualCoOccurence.CandidateGenerationTextualCoOccurence("../data/" + directory + "/userEmbeddings600.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"",
"",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_COOCCURENCE,
"../data/" + directory, 
15,
datasetName)
cgtTexCo.runNeuralNetwork()