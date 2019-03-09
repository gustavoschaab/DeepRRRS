import sys
sys.path.append('../')
from analysis import data_analysis

# da = data_analysis.DatasetAnalysis("../data/AmazonClothings/meta_Clothing_Shoes_and_Jewelry.json", 
#                                 "../data/AmazonClothings/ratings_5_core.csv",
#                                 "reports/AmazonClothings",
#                                 "Amazon Clothings, Shoes and Jewerly")
# da.run()

# da = data_analysis.DatasetAnalysis("../data/AmazonHealth/meta_Health.json", 
#                                 "../data/AmazonHealth/ratings_5_core_health.csv",
#                                 "reports/AmazonHealth",
#                                 "Amazon Health")
# da.run()

da = data_analysis.DatasetAnalysis("../data/AmazonHome/meta_Home.json", 
                                "../data/AmazonHome/ratings_5_core_home.csv",
                                "reports/AmazonHome",
                                "Amazon Home")
da.run()