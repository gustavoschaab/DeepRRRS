import sys
sys.path.append('../')
from modules.candidate_generation import candidateGeneration

#Only Textual Embeddings
cg = candidateGeneration.CandidateGeneration("../data/AmazonClothings/userEmbeddings300.pkl", 
"../data/AmazonClothings/textual_embeddings.pkl", 
"../data/AmazonClothings/ratings_5_core.csv", 
"../data/AmazonClothings/model_candidate_generation600_1.h5", 
70,
"CLOTHINGS TEXTUAL")
cg.runNeuralNetwork()

