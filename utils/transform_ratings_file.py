import json
import csv
import ast

class TranformFile(object):

    def __init__(self, inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile

    def transformFile(self):
        products = []

        with open (self.inputFile, 'rt') as in_file:
            for line in in_file:
                product = ast.literal_eval(line)
                lst = []
                lst.append(product['reviewerID'])
                lst.append(product['asin'])
                lst.append(product['overall'])

                products.append(lst)

        with open(self.outputFile, 'w', newline='') as myfile:
            wr = csv.writer(myfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            for value in products:
                wr.writerow(value)