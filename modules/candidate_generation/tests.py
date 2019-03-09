import csv

lst = []
with open("D:/git/master_degree_model/modules/candidate_generation/ratings_5_core.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    lst = list(reader)
new_lst = [item[1] for item in lst if item[0] == "A1KLRMWW2FWPL4"]
print(len(new_lst))
print(new_lst)
