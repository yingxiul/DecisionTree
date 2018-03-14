import sys
import os
import csv
import math

def inspect(input,output):
    
    total = 0
    d = dict()
    entropy = 0
    erro = 0
    minority = -1
    
    with open(input,'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        #skip the field name
        next(csv_reader)
        
        for line in csv_reader:
            total += 1
            if (line[-1] in d):
                d[line[-1]] += 1
            else:
                d[line[-1]] = 1
        
        for key in d:
            
            if (minority == -1):
                minority = d[key]
            else:
                minority = min(minority, d[key])
            
            val = d[key]/float(total)
            entropy += - val * math.log(val,2)
       
    str = "entropy: {}\nerro: {}".format(entropy,minority/float(total))
    
    outfile = open(output,"w")
    outfile.write(str)
    
    csv_file.close()
    outfile.close()

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    
    inspect(input,output)