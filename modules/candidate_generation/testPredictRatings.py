import predictRatings
import pickle

pr = predictRatings.PredictRatings("A1KLRMWW2FWPL4", "model_candidate_generation.h5", "users_dense_vector.pkl", "../../data/textual_embeddings.pkl", "ratings_5_core.csv")
#p = pr.runPredictions()
#with open("predictions.out", 'wb') as f:
#    pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)

with open("predictions.out", 'rb') as f:
    p = pickle.load(f)

p_calc = pr.convertPredictions(p)
with open("predictions_calculated.out", 'wb') as f:
    pickle.dump(p_calc, f, pickle.HIGHEST_PROTOCOL)