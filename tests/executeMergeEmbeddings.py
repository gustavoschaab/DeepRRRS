import sys
sys.path.append('../')
from modules.merge_embeddings import mergeEmbeddings

# me = mergeEmbeddings.MergeEmbeddings("../data/AmazonClothings/textual_embeddings.pkl",
# "../data/AmazonClothings/categoricalEmbeddings.pkl",
# "",
# "",
# "../data/AmazonClothings/mergedEmbeddings.pkl")
# me.merge()

me = mergeEmbeddings.MergeEmbeddings("",
"../data/AmazonClothings/categoricalEmbeddings.pkl",
"",
"../data/AmazonClothings/coOccurenceEmbeddings.pkl",
"../data/AmazonClothings/categoricalCoOccurenceEmbeddings.pkl",
"CLOTHINGS - CATEGORICAL + CO-OCCURENCE")
me.merge()