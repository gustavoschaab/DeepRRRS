import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration_Image
from modules.candidate_generation import candidateGeneration_TextualImage
from modules.candidate_generation import candidateGeneration_TextualCategoricalCoOccurenceImage
from modules.candidate_generation import candidateGeneration_TextualCategoricalImage
from modules.candidate_generation import candidateGeneration_TextualCoOccurenceImage
from modules.candidate_generation import candidateGeneration_CategoricalCoOccurenceImage
from modules.candidate_generation import candidateGeneration_CategoricalImage
from modules.candidate_generation import candidateGeneration_CoOccurenceImage
from modules.candidate_generation import featureCombination
from modules.feature_extraction import generateUserEmbeddings
from modules.feature_extraction import imageFeature

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181208_clothings_image.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

directory = "AmazonHealth"
csfFile = "ratings_5_core.csv"
metaFile = "meta_health.json"
datasetName = "Health"

#Generate User Embeddings with 4096 dimensions
u4096v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
"../data/" + directory + "/userEmbeddings4096.pkl",
4096)
u4096v.generateEmbeddings()

#Generate User Embeddings with 4396 dimensions
u4396v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
"../data/" + directory + "/userEmbeddings4396.pkl",
4396)
u4396v.generateEmbeddings()

#Generate User Embeddings with 4696 dimensions
u4696v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
"../data/" + directory + "/userEmbeddings4696.pkl",
4696)
u4696v.generateEmbeddings()

#Generate User Embeddings with 4996 dimensions
u4996v = generateUserEmbeddings.User2Vec("../data/" + directory + "/" + csfFile,
"../data/" + directory + "/userEmbeddings4996.pkl",
4996)
u4996v.generateEmbeddings()

#Extract image features and generates embeddings with 4096 dimensions
imageFeature = imageFeature.ImageFeature("../data/" + directory + "/image_features.b", 
"../data/" + directory + "/imageEmbeddings.pkl",
"HOME - IMAGE EMBEDDINGS",
 "../data/" + directory + "/" + csfFile,
 False)
imageFeature.generateEmbeddings()

#Neural Network Image
nn1 = candidateGeneration_Image.CandidateGenerationImage("../data/" + directory + "/userEmbeddings4096.pkl", 
"", 
"../data/" + directory + "/imageEmbeddings.pkl",
"",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.IMAGE,
"../data/" + directory, 
15,
datasetName)
nn1.runNeuralNetwork() 

#Neural Network Textual + Image
nn2 = candidateGeneration_TextualImage.CandidateGenerationTextualImage("../data/" + directory + "/userEmbeddings4396.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"../data/" + directory + "/imageEmbeddings.pkl",
"",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn2.runNeuralNetwork()

#Neural Network Textual + Categorical + Co-Occurence + Image
nn3 = candidateGeneration_TextualCategoricalCoOccurenceImage.CandidateGenerationTextualCategoricalCoOccurenceImage(
"../data/" + directory + "/userEmbeddings4996.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"../data/" + directory + "/imageEmbeddings.pkl",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_CATEGORICAL_COOCCURENCE_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn3.runNeuralNetwork()

#Neural Network Textual + Categorical + Image
nn4 = candidateGeneration_TextualCategoricalImage.CandidateGenerationTextualCategoricalImage(
"../data/" + directory + "/userEmbeddings4696.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"../data/" + directory + "/imageEmbeddings.pkl",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_CATEGORICAL_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn4.runNeuralNetwork()

#Neural Network Textual + Co-Occurence + Image
nn5 = candidateGeneration_TextualCoOccurenceImage.CandidateGenerationTextualCoOccurenceImage(
"../data/" + directory + "/userEmbeddings4696.pkl", 
"../data/" + directory + "/textual_embeddings.pkl", 
"../data/" + directory + "/imageEmbeddings.pkl",
"",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.TEXTUAL_COOCCURENCE_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn5.runNeuralNetwork()

#Neural Network Categorical + Co-Occurence + Image
nn6 = candidateGeneration_CategoricalCoOccurenceImage.CandidateGenerationCategoricalCoOccurenceImage(
"../data/" + directory + "/userEmbeddings4696.pkl", 
"", 
"../data/" + directory + "/imageEmbeddings.pkl",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.CATEGORICAL_COOCCURENCE_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn6.runNeuralNetwork()

#Neural Network Categorical + Image
nn7 = candidateGeneration_CategoricalImage.CandidateGenerationCategoricalImage(
"../data/" + directory + "/userEmbeddings4396.pkl", 
"", 
"../data/" + directory + "/imageEmbeddings.pkl",
"../data/" + directory + "/categoricalEmbeddings300.pkl",
".",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.CATEGORICAL_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn7.runNeuralNetwork()

#Neural Network Co-Occurence + Image
nn8 = candidateGeneration_CoOccurenceImage.CandidateGenerationCoOccurenceImage(
"../data/" + directory + "/userEmbeddings4396.pkl", 
"", 
"../data/" + directory + "/imageEmbeddings.pkl",
"",
"../data/" + directory + "/coOccurenceEmbeddings300.pkl",
"../data/" + directory + "/" + csfFile, 
featureCombination.EnumFeatureCombination.COOCCURENCE_IMAGE,
"../data/" + directory, 
15,
datasetName)
nn8.runNeuralNetwork()