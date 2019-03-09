import rankingGeneration
import pickle

with open("../candidate_generation/predictions_calculated.out", 'rb') as f:
    pred = pickle.load(f)

rg = rankingGeneration.RankingGeneration(pred, 20)
topN = rg.getTopN()

for item in topN:
    print("Item: " + str(item[0]) + " Rating: " + str(item[1]))
