import sys
import logging
sys.path.append('../')
from modules.candidate_generation import candidateGeneration_cooccurence
from modules.candidate_generation import featureCombination
from modules.feature_extraction import coOccurenceFeature

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='./logs/execution_20181127_clothings.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# cf = coOccurenceFeature.CoOccurenceFeature(    
# "../data/AmazonHome/meta_Home.json",
# "../data/AmazonHome/coOccurenceEmbeddings.pkl",
# "../data/AmazonHome/coOccurenceEmbeddings.model")
# cf.generateCoOccurenceEmbeddings()

cgc = candidateGeneration_cooccurence.CandidateGenerationCooccurence("../data/AmazonClothings/userEmbeddings300.pkl", 
"", 
"",
"",
"../data/AmazonClothings/coOccurenceEmbeddings300.pkl",
"../data/AmazonClothings/ratings_5_core.csv", 
featureCombination.EnumFeatureCombination.COOCCURENCE,
"../data/AmazonClothings", 
15,
"Clothings")
cgc.runNeuralNetwork()