import sys
sys.path.append('../')
from modules.candidate_generation import candidateGeneration

cg = candidateGeneration.CandidateGeneration("../data/AmazonClothings/userEmbeddings300.pkl", 
"../data/AmazonClothings/textual_embeddings.pkl", 
"../data/AmazonClothings/ratings_5_core.csv", 
"../data/AmazonClothings/model_candidate_generation600.h5", 70,
"CLOTHINGS = USER 300")
cg.calculateMetrics()


#from sklearn.metrics import mean_absolute_error
#expected = [0.0, 0.5, 0.0, 0.5, 0.0]
#predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
""" mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)

expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.0, 0.5, 0.0, 0.5, 0.0]
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)

expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.5, 0.9, 0.6, 0.1, 0.7]
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)

expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [1.0, 1.5, 1.0, 1.5, 1.0]
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae)

expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.01, 0.5, 0.0, 0.5, 0.0]
mae = mean_absolute_error(expected, predictions)
print('MAE: %f' % mae) """