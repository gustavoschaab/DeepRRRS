import sys
sys.path.append('../')
from modules.candidate_generation import candidateGeneration2
from modules.candidate_generation import featureCombination

#Only Textual Embeddings
# cg = candidateGeneration2.CandidateGeneration("../data/AmazonClothings/userEmbeddings300.pkl", 
# "../data/AmazonClothings/textual_embeddings.pkl", 
# "",
# "",
# "",
# "../data/AmazonClothings/ratings_5_core.csv", 
# featureCombination.EnumFeatureCombination.TEXTUAL,
# "../data/AmazonClothings", 
# 7,
# "Clothing, Shoes and Jewelry")
# cg.runNeuralNetwork()

cg = candidateGeneration2.CandidateGeneration("../data/AmazonClothings/userEmbeddings300.pkl", 
"../data/AmazonClothings/textual_embeddings.pkl", 
"",
"",
"",
"../data/AmazonClothings/ratings_5_core.csv", 
featureCombination.EnumFeatureCombination.TEXTUAL,
"../data/AmazonClothings", 
15,
"AmazonClothings")
cg.runNeuralNetwork()