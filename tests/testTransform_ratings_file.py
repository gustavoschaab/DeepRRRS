import sys
sys.path.append('../')
from utils import transform_ratings_file

tf = transform_ratings_file.TranformFile("../data/AmazonElectronics/reviews_electronics.json",
"../data/AmazonElectronics/ratings_5_core.csv")
tf.transformFile()