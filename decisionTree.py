import sys
import os
import csv
import math


def train_and_test(train_input,test_input,depth,train_out,test_out,metrics):
    data = handle_data(train_input)
    train_label = data[0]
    train_feats = data[1]
    train_tags = data[2]
    
    tree = decisionTreeTrain(train_label, train_feats, train_tags, 
                             0, int(depth))
    #printTree(tree,0)
    
    train_erro = test(tree, train_input, train_out)
    test_erro = test(tree, test_input, test_out)
    
    str = "error(train): {}\nerror(test): {}".format(train_erro, test_erro)
    f = open(metrics, 'w')
    f.write(str)
    
    f.close()

##
## Helper functions

# handle_data: takes in the input csv file. Returns last column as labels,
# features with values as dict and a tag list for labels.
def handle_data(input):
    label = list()
    feats = list()
    d = dict()
    
    with open(input,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for line in csv_reader:
            l = len(line)
            
            if (len(feats) == 0):
                for i in range(l-1):
                    feats.append([line[i]])
            else:
                label.append(line[-1])
                for i in range(l-1):
                    feats[i].append(line[i])
        
        feats_num = len(feats)
        for j in range(feats_num):
            feat = feats[j].pop(0)
            d[feat] = feats[j]
    
    csv_file.close()
    
    tags = list(set(label))
    counts = count_labels(label, tags)
    print("[{} {} /{} {}]".format(counts[0],tags[0],counts[1],tags[1]))
    
    return [label,d,tags]

# decisionTreeTrain: build the decision tree
def decisionTreeTrain(labels, feats, tags, cur_depth, max_depth):
    tag0_num = count_labels(labels,tags)[0]
    tag1_num = count_labels(labels,tags)[1]
    
    guess = tags[0]                     #get the majority label
    guess_num = tag0_num                #get the majority label's num
    
    if (tag1_num > tag0_num):
        guess = tags[1]                        
        guess_num = tag1_num
                 
    if (guess_num == len(labels)):      #base case: no need to split further
        return Tree(guess)
    elif (len(feats) == 0):
        return Tree(guess)              #base case: cannot split further
    elif (cur_depth >= max_depth):
        return Tree(guess)
    else:
        n = list()
        y = list()
        n_tag = ''
        y_tag = ''
        score = -1
        split = ''
        
        #loop through all remaining features and pick the one with most info
        for key in feats:
            info = info_gain(labels,feats[key])
            cur_n = info[0]
            cur_y = info[1]
            cur_score = info[2]
            tag1 = info[3]
            tag2 = info[4]
            
            if (score == -1 or score < cur_score):
                score = cur_score
                n = cur_n
                y = cur_y
                n_tag = tag1
                y_tag = tag2
                split = key
        
        # case when x has multi values but y has only one possible value
        if (y_tag == None):
            return Tree(guess)
        
        cur_depth += 1
        cur_feat = feats[split]
        cur_tag = [n_tag, y_tag]
        del feats[split]
        new_feats = split_feats(feats, cur_feat, cur_tag)
        
        n_count = count_labels(n,tags)
        print('| '*cur_depth,split,' = ',n_tag,': [',n_count[0],' ',tags[0],'/',
              n_count[1],' ',tags[1],']')
        
        left = decisionTreeTrain(n,new_feats[0], tags, cur_depth, max_depth)
        
        y_count = count_labels(y,tags)
        print('| '*cur_depth,split,' = ',y_tag,': [',y_count[0],' ',tags[0],'/',
              y_count[1],' ',tags[1],']')
        
        right = decisionTreeTrain(y,new_feats[1], tags, cur_depth, max_depth)
        
        return Tree(split, left,right,n_tag)

# test the result of the tree, returns the error rate
def test(tree, input, output):
    feat = list()
    d = dict()
    data = list()
    count = 0
    total = 0
    
    with open(input,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for line in csv_reader:
            l = len(line)
            total += 1
            
            if (len(feat) == 0):
                for i in range(l-1):
                    feat.append(line[i])
            else:
                for i in range(l-1):
                    d[feat[i]] = line[i]
                label = decisionTreeTest(tree, d)
                data.append(label + '\n')
                if (label != line[-1]):
                    count += 1
    
    str = "".join(data)
    f = open(output,'w')
    f.write(str)
    
    csv_file.close()
    f.close()
    
    return float(count) / (total - 1)
    
# decisionTreeTest takes in the decision tree and a dict test point
# returns the label
def decisionTreeTest(tree, d):
    if (tree.isLeaf()):
        return tree.feature
    else:
        if (d[tree.feature] == tree.split):
            return decisionTreeTest(tree.left, d)
        else:
            return decisionTreeTest(tree.right, d)

##
# coun_labels take in the labels and tags as lists and return a list, 
# [tag1_num, tag2_num]
# the len of tags should be 2
def count_labels(labels, tags):
    l = [0, 0]

    for label in labels:
        if (label == tags[0]):
            l[0] += 1
        else:
            l[1] += 1
    
    return l

# info_gain: calculate info_gain I(y,x) = H(y) - H(y|x)
def info_gain(labels,feat):
    N = list()
    Y = list()
    neg = 0
    pos = 0
    
    tag = list(set(feat))
    label = list(set(labels))
    H = get_entropy(labels, label)
    
    if (len(tag) == 1):
        score = H - get_entropy(labels,label)
        return [labels,[],score,tag[0],None]
    else:
        for i in range(len(feat)):
            if (feat[i] == tag[0]):
                neg += 1
                N.append(labels[i])
            else:
                pos += 1
                Y.append(labels[i])
        
        score = H - (float(neg) / len(labels) * get_entropy(N,label)
                +float(pos) / len(labels) * get_entropy(Y,label))
        #print(score)
        return [N,Y,score,tag[0],tag[1]]

# get_entropy: H(y|x) = -1 * sum(P(Y=y|X=x)*logP(Y=y|X=x))
def get_entropy(labels,tag):
    n_num = count_labels(labels,tag)[0]
    y_num = len(labels) - n_num
    
    if (len(set(labels)) == 1):
        return 0
    
    a = float(n_num) / len(labels)
    b = float(y_num) / len(labels)

    entro = -1 * (a * math.log(a , 2) +  b * math.log(b , 2))
    
    return entro

# split input feats into two feats based on the given source
def split_feats(feats, source, tag):
    n_d = dict()
    y_d = dict()
    
    for key in feats:
        n_d[key] = list()
        y_d[key] = list()
        for i in range(len(source)):
            if (source[i] == tag[0]):
                n_d[key].append(feats[key][i])
            else:
                y_d[key].append(feats[key][i])
    
    return [n_d,y_d]

# printTree: print out the decision tree
def printTree(tree,depth):
    if (tree.isLeaf()):
        print(' '*depth,tree.feature)
    else:
        print(' '*depth,tree.feature,': ',tree.split)
        printTree(tree.left,depth+1)
        printTree(tree.right,depth+1)
##
## Tree class
class Tree(object):
    def __init__(self, feature, left = None, right = None, split = None):
        self.feature = feature
        self.left = left
        self.right = right
        self.split = split           #tag for left brance
    
    def isLeaf(self):
        if (self.left == None and self.right == None):
            return True
        else:
            return False
    
##
## Main function
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    depth = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
        
    train_and_test(train_input,test_input,depth,train_out,test_out,metrics)