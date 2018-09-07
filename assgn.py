import numpy as np
import math as ma
import random
from random import randint
import copy
import sys
import pickle

#################################################################################################################
# CLASS TREE NODE
#################################################################################################################

global no_of_node
global max_no_node
global FrequencyOfAttributes
FrequencyOfAttributes = [0]*5000


class Node:
    def __init__(self):
        self.classify = None
        self.classEntropy = 1
        self.entropy = 1
        self.infoSame = 0
        self.infoZero = 0
        self.index = -1
        self.left = None
        self.parent = None
        self.right = None
        self.dataset = None
        self.positive = None
        self.ypos = None
        self.isLeaf = False
        self.IsPossible = False
        self.negative = None
        self.yneg = None
        self.checked = False

    def __copy__(self):
        a = Node()
        a.classify = self.classify
        a.classEntropy = self.classEntropy
        a.entropy = self.entropy
        a.infoSame = self.infoSame
        a.infoZero = self.infoZero
        a.index = self.index
        a.left = self.left
        a.checked = self.checked
        a.right = self.right
        a.parent = self.parent
        a.dataset = self.dataset
        a.positive = self.positive
        a.ypos = self.ypos
        a.isLeaf = self.isLeaf
        a.negative = self.negative
        a.IsPossible = self.IsPossible
        a.yneg = self.yneg
        return a


#################################################################################################################
# FUNCTION TO ID3
#################################################################################################################


def ID3(dataset, feature):
    global no_of_node
    global max_no_node
    if (no_of_node >= max_no_node):
        root = Node()
        a = np.count_nonzero(feature)
        b = len(feature) - a
        if (a > b):
            root.classify = 1
 #           root.IsPossible = True
            root.isLeaf = True
 #           root.checked = True
            return root
        else:
            root.classify = 0
 #           root.IsPossible = True
 #           root.checked = True
            root.isLeaf = True
            return root
    no_of_node = no_of_node + 1
    #print(no_of_node)
    if (not np.isin(0, feature)):
        root = Node()
 #       root.IsPossible = True
        root.isLeaf = True
#        root.checked = True
        root.classify = 1
        return root
    if (not np.isin(1, feature)):
        root = Node()
 #       root.IsPossible = True
        root.isLeaf = True
 #       root.checked = True
        root.classify = 0
        return root

    root = INFOGAIN(dataset, feature)
    if (root.infoZero == 1):
        a = root.positive
        b = root.negative
        if (a > b):
#            root.IsPossible = True
            root.isLeaf = True
            root.classify = 1
#            root.checked = True
            return root
        else:
 #           root.IsPossible = True
            root.isLeaf = True
#            root.checked = True
            root.classify = 0
            return root
    else:
        X_pos = []
        X_neg = []
        y_pos = []
        y_neg = []

        for key, value in zip(root.ypos, root.positive):
            X_pos.append(value)
            y_pos.append(key)

        for key, value in zip(root.yneg, root.negative):
            X_neg.append(value)
            y_neg.append(key)

        root.left = ID3(X_neg, y_neg)
  #      root.left.parent = root

        root.right = ID3(X_pos, y_pos)
 #       root.right.parent = root

    return root


#################################################################################################################
# FUNCTION TO CALCULATE INFORMATION GAIN
#################################################################################################################

def INFOGAIN(dataset, y):
    global FrequencyOfAttributes
    node = Node()
    InfoGainVector = []
    for i in range(len(dataset[0])):
        pp = [item[i] for item in dataset]
        InfoGainVector.append(Entropy(y) - Entropy_info(pp, y))

    InfoGainVector = np.array(InfoGainVector)
    uniqueelement = np.unique(InfoGainVector)

    if (len(uniqueelement) > 1):
        localindex = np.argmax(InfoGainVector)
        node.index = localindex
        FrequencyOfAttributes[localindex] = 1 + FrequencyOfAttributes[localindex]
        positivelist = []
        yposlist = []
        negativelist = []
        yneglist = []

        for key, value in zip(y, dataset):
            if (value[localindex] > 0):
                positivelist.append(value)
                yposlist.append(key)
            else:
                negativelist.append(value)
                yneglist.append(key)

        node.positive = positivelist
        node.negative = negativelist
        node.ypos = yposlist
        node.yneg = yneglist
    else:
        if (InfoGainVector[0] == 0):
            node.infoZero = 1
            positivelist = []
            yposlist = []
            negativelist = []
            yneglist = []

            for key, value in zip(y, dataset):
                if (key > 0):
                    positivelist.append(value)
                    yposlist.append(key)
                else:
                    negativelist.append(value)
                    yneglist.append(key)

            node.positive = positivelist
            node.negative = negativelist
            node.ypos = yposlist
            node.yneg = yneglist

        else:
            node.infoSame = 1
            localindex = np.argmax(InfoGainVector)
            node.index = localindex
            FrequencyOfAttributes[localindex] = 1 + FrequencyOfAttributes[localindex]
            positivelist = []
            yposlist = []
            negativelist = []
            yneglist = []

            for key, value in zip(y, dataset):
                if (value[localindex] > 0):
                    positivelist.append(value)
                    yposlist.append(key)
                else:
                    negativelist.append(value)
                    yneglist.append(key)

            node.positive = positivelist
            node.negative = negativelist
            node.ypos = yposlist
            node.yneg = yneglist

    return node


#################################################################################################################
# FUNCTION TO CALCULATE ENTROPY
#################################################################################################################


def Entropy_info(dataset, y):
    no_of_example = len(y)
    poscount_pos = 0
    poscount_neg = 0
    negcount_pos = 0
    negcount_neg = 0

    for i, j in zip(dataset, y):
        if (i > 0 and j > 0):
            poscount_pos = poscount_pos + 1
        elif (i > 0 and j <= 0):
            poscount_neg = poscount_neg + 1
        elif (i <= 0 and j > 0):
            negcount_pos = negcount_pos + 1
        elif (i <= 0 and j <= 0):
            negcount_neg = negcount_neg + 1

    if (poscount_pos == 0 or poscount_neg == 0):
        entropy1 = 0
    else:
        entropy1 = (-poscount_pos / (poscount_pos + poscount_neg)) * (
            ma.log(poscount_pos / (poscount_pos + poscount_neg), 2)) + (
                           (-poscount_neg / (poscount_pos + poscount_neg)) * (
                       ma.log(poscount_neg / (poscount_pos + poscount_neg), 2)))

    if (negcount_pos == 0 or negcount_neg == 0):
        entropy2 = 0
    else:
        entropy2 = (-negcount_pos / (negcount_pos + negcount_neg)) * (
            ma.log(negcount_pos / (negcount_pos + negcount_neg), 2)) + (
                           (negcount_neg / (negcount_pos + negcount_neg)) * (
                       -ma.log(negcount_neg / (negcount_neg + negcount_pos), 2)))

    entropy = ((poscount_pos + poscount_neg) / no_of_example) * entropy1 + (
            (negcount_pos + negcount_neg) / no_of_example) * entropy2
    return entropy


#################################################################################################################
# HELPING FUNCTION OF ENTROPY OF Y
#################################################################################################################


def Entropy(Y_out):
    y = np.array(Y_out)
    res = np.bincount(y)

    if (res[0] == 0 or res[1] == 0):
        return 0
    else:
        return (res[0] / (res[0] + res[1])) * (-ma.log(res[0] / (res[0] + res[1]), 2)) + (res[1] / (res[0] + res[1])) * (-ma.log(res[1] / (res[0] + res[1]), 2))


#################################################################################################################
# FUNCTION TO CALCULATE TREE ACCURACY
#################################################################################################################


def accuracy(root, dataset, y):
    y_predict = []
    temp = copy.copy(root)
    for inst in dataset:
        y_predict.append(predict(temp, inst))

    count = 0
    for i in range(len(y)):
        if (y[i] == y_predict[i]):
            count = count + 1

    acc = count * 100 / len(y)
    return acc


#################################################################################################################
# FUNCTION TO PREDICT ON INSTANCE
#################################################################################################################


def predict(root, instance):
    temp = copy.copy(root)
    while (True):
        if (not temp.left):
            return temp.classify
        if (instance[temp.index] > 0):
            temp = temp.right
        else:
            temp = temp.left


def printInorder(root):
    if root:
        printInorder(root.left)
        print(root.classify)
        printInorder(root.right)


#################################################################################################################
# FUNCTION TO LOAD TRAIN DATA
#################################################################################################################

def loadTrainDataset():
    No_of_feature = 5000
    No_of_Instance = 1000
    GlobalMapping = createMapping()
    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('MyTrainData.txt', 'r') as myfile:
        line = myfile.readline().strip()
        cnt = 0
        noofpf = 1
        noofnf = 1
        while (line):
            if ((int(line[0:2]) >= 7)):
                TrainDatasetP[cnt] = line
                noofpf = noofpf + 1
            if (int(line[0:2]) <= 4):
                TrainDatasetN[cnt] = line
                noofnf = noofnf + 1
            cnt += 1
            line = myfile.readline().strip()
    myfile.close()

    TrainDatasetP = random.sample(TrainDatasetP.items(),len(TrainDatasetP.items()))
    TrainDatasetN = random.sample(TrainDatasetN.items(),len(TrainDatasetN.items()))

    TrainDatasetP = TrainDatasetP + TrainDatasetN

    MapIndexToFeat = GlobalMapping.copy()
    FeatureMatrix = []
    Pex = int(No_of_Instance / 2)
    y_out = [0] * No_of_Instance
    inst = 0
    for key,value in TrainDatasetP:
        InstRow = [0] * (No_of_feature + 1)
        featu = value.split(" ")
        skip = 0
        if (Pex):
            y_out[inst] = 1
            InstRow[No_of_feature] = 1
            Pex = Pex - 1
        for feature in featu:
            if (skip == 0):
                skip = 1
            else:
                if (int(feature.split(':')[0]) in MapIndexToFeat.keys()):
                    # InstRow[MapIndexToFeat[int(feature.split(":")[0])]]  = FileExp[int(feature.split(":")[0])];
                    InstRow[MapIndexToFeat[int(feature.split(":")[0])]] = int(feature.split(':')[1])
        FeatureMatrix.append(InstRow)
        inst = inst + 1
    return FeatureMatrix


#################################################################################################################
# FUNCTION TO LOAD TEST DATA
#################################################################################################################

def loadTestDataset():
    GlobalMapping = createMapping()
    No_of_feature = 5000
    No_of_Instance = 1000

    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('MyTestData.txt', 'r') as myfile:
        line = myfile.readline().strip()
        cnt = 0
        noofpf = 1
        noofnf = 1
        while (line):
            if ((int(line[0:2]) >= 7)):
                TrainDatasetP[cnt] = line
                noofpf = noofpf + 1
            if (int(line[0:2]) <= 4):
                TrainDatasetN[cnt] = line
                noofnf = noofnf + 1
            cnt += 1
            line = myfile.readline().strip()
    myfile.close()

    TrainDatasetP = random.sample(TrainDatasetP.items(), len(TrainDatasetP.items()))
    TrainDatasetN = random.sample(TrainDatasetN.items(), len(TrainDatasetN.items()))

    TrainDatasetP = TrainDatasetP + TrainDatasetN


    MapIndexToFeat = GlobalMapping.copy()

    FeatureMatrix = []
    Pex = int(No_of_Instance / 2)
    y_out = [0] * No_of_Instance
    inst = 0
    for key,value in TrainDatasetP:
        #    	print(value)
        InstRow = [0] * (No_of_feature + 1)
        featu = value.split(" ")
        skip = 0
        cnt = 0
        if (Pex):
            y_out[inst] = 1
            InstRow[No_of_feature] = 1
            Pex = Pex - 1
        for feature in featu:
            if (skip == 0):
                skip = 1
            else:
                if (int(feature.split(':')[0]) in MapIndexToFeat.keys()):
                    # InstRow[MapIndexToFeat[int(feature.split(":")[0])]]  = FileExp[int(feature.split(":")[0])];
                    InstRow[MapIndexToFeat[int(feature.split(":")[0])]] = int(feature.split(':')[1])
        FeatureMatrix.append(InstRow)
        inst = inst + 1
    return FeatureMatrix

#################################################################################################################
# FUNCTION TO BUILD TREE
#################################################################################################################

def buildTree(dataset):
    global FrequencyOfAttributes
    global no_of_node
    no_of_node = 0
    y_out = [item[-1] for item in dataset]
    FeatureMatrix = [item[0:-1] for item in dataset]
    root = ID3(FeatureMatrix, y_out)
    return root


#################################################################################################################
# FUNCTION TO CALCUALTEDECISION TREE ACCURACY
#################################################################################################################

def DecisionTreeAccuracy(root, dataset):
    temp = copy.copy(root)
    y_out = [item[-1] for item in dataset]
    FeatureMatrix = [item[0:-1] for item in dataset]
    acc = accuracy(temp, FeatureMatrix, y_out)
    return acc


#################################################################################################################
# FUNCTION TO CALCUALTE RANDOM FOREST ACCURACY
#################################################################################################################


def RandomAccuracy(Trees, dataset):
    y_out = [item[-1] for item in dataset]
    FeatureMatrix = [item[0:-1] for item in dataset]
    TreePrediction = []
    ForestPredict = []
    for root in Trees:
        y_predict = []
        for inst in FeatureMatrix:
            y_predict.append(predict(root, inst))
        TreePrediction.append(y_predict)

    for i in range(len(TreePrediction[0])):
        y = [item[i] for item in TreePrediction]
        ones = np.count_nonzero(y)
        zeros = len(y) - ones
        if (ones > zeros):
            ForestPredict.append(1)
        else:
            ForestPredict.append(0)

    acc = sum(1 for x, y in zip(ForestPredict, y_out) if x == y) / len(y_out)
    return acc*100


#################################################################################################################
# FUNCTION TO BUILD FOREST Random Tree
#################################################################################################################

def RandomTree(number):
    GlobalMapping = createMapping()
    global max_no_node
    global no_of_node
    No_of_tree = []
    No_of_Instance = 1000
    No_of_feature = 5000
    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('MyTrainData.txt', 'r') as myfile:
        line = myfile.readline().strip()
        cnt = 0
        noofpf = 1
        noofnf = 1
        while (line):
            if ((int(line[0:2]) >= 7)):
                TrainDatasetP[cnt] = line
                noofpf = noofpf + 1
            if (int(line[0:2]) <= 4):
                TrainDatasetN[cnt] = line
                noofnf = noofnf + 1
            cnt += 1
            line = myfile.readline().strip()
    myfile.close()

    TrainDatasetP = random.sample(TrainDatasetP.items(), len(TrainDatasetP.items()))
    TrainDatasetN = random.sample(TrainDatasetN.items(), len(TrainDatasetN.items()))

    TrainDatasetP = TrainDatasetP + TrainDatasetN

    TotalVocab = GlobalMapping.copy()
   # print(GlobalMapping)
    for i in range(number):
        PosNegIndex = random.sample(TotalVocab.items(), 1000)
        MapIndexToFeat = {}
        for key,value in PosNegIndex:
            MapIndexToFeat[key] = GlobalMapping[key]

        FeatureMatrix = []
        Pex = int(No_of_Instance / 2)
        y_out = [0] * No_of_Instance
        inst = 0
        for key, value in TrainDatasetP:
            InstRow = [0] * (No_of_feature )
            featu = value.split(" ")
            skip = 0
            if (Pex):
                y_out[inst] = 1
                Pex = Pex - 1
            for feature in featu:
                if (skip == 0):
                    skip = 1
                else:
                    if (int(feature.split(':')[0]) in MapIndexToFeat.keys()):
                        # InstRow[MapIndexToFeat[int(feature.split(":")[0])]]  = FileExp[int(feature.split(":")[0])];
                        InstRow[MapIndexToFeat[int(feature.split(":")[0])]] = int(feature.split(':')[1])
            FeatureMatrix.append(InstRow)
            inst = inst + 1
        no_of_node = 0
        max_no_node = 3000
        root = ID3(FeatureMatrix, y_out)
        No_of_tree.append(root)
    return No_of_tree


#################################################################################################################
# FUNCTION TO ADD NOISE
#################################################################################################################

def addNoise(dataset, per):
    global FrequencyOfAttributes
    global max_no_node
    global no_of_node
    no_of_node = 0
    max_no_node = 1700
    y_out = [item[-1] for item in dataset]
    FeatureMatrix = [item[0:-1] for item in dataset]

    datalength = len(y_out)

    looptimes = int((per * datalength) / 100)

    for i in range(looptimes):
        randomindex = np.random.randint(0, len(y_out))
        if (y_out[randomindex] == 1):
            y_out[randomindex] = 0
        else:
            y_out[randomindex] = 1

    root = ID3(FeatureMatrix, y_out)
    return root

#################################################################################################################
# Create Mapping
#################################################################################################################


def createMapping():
    GlobalMapping = {}
    PosIndex = []
    NegIndex = []
    with open('MyVocab.txt', 'r') as myfile:
        line = myfile.readline()
        count = 0
        while (line):
            if (count < 2500):
                PosIndex.append(int(line))
                count += 1
            else:
                NegIndex.append(int(line))
            line = myfile.readline()
    myfile.close()
    index = 0
    for word in PosIndex:
        GlobalMapping[word] = index
        index = index + 1
    for word in NegIndex:
        GlobalMapping[word] = index
        index = index + 1
    # print('Mapping is created')
    # print(GlobalMapping)
    return  GlobalMapping

#################################################################################################################
# Create Validation Set
#################################################################################################################

def validationSet():
    GlobalMapping = createMapping()
    No_of_feature = 5000
    No_of_Instance = 1000
    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('MyTestData.txt', 'r') as myfile:
        line = myfile.readline().strip()
        cnt = 0
        noofpf = 1
        noofnf = 1
        while (line):
            if ((int(line[0:2]) >= 7)):
                TrainDatasetP[cnt] = line
                noofpf = noofpf + 1
            if (int(line[0:2]) <= 4):
                TrainDatasetN[cnt] = line
                noofnf = noofnf + 1
            cnt += 1
            line = myfile.readline().strip()
    myfile.close()

    TrainDatasetP = random.sample(TrainDatasetP.items(), int(len(TrainDatasetP.items())/3))
    TrainDatasetN = random.sample(TrainDatasetN.items(), int(len(TrainDatasetN.items())/3))

    TrainDatasetP = TrainDatasetP + TrainDatasetN


    MapIndexToFeat = GlobalMapping.copy()

    FeatureMatrix = []
    Pex = int(No_of_Instance / 2)
    y_out = [0] * No_of_Instance
    inst = 0
    for key,value in TrainDatasetP:
        #    	print(value)
        InstRow = [0] * (No_of_feature + 1)
        featu = value.split(" ")
        skip = 0
        cnt = 0
        if (Pex):
            y_out[inst] = 1
            InstRow[No_of_feature] = 1
            Pex = Pex - 1
        for feature in featu:
            if (skip == 0):
                skip = 1
            else:
                if (int(feature.split(':')[0]) in MapIndexToFeat.keys()):
                    # InstRow[MapIndexToFeat[int(feature.split(":")[0])]]  = FileExp[int(feature.split(":")[0])];
                    InstRow[MapIndexToFeat[int(feature.split(":")[0])]] = int(feature.split(':')[1])
        FeatureMatrix.append(InstRow)
        inst = inst + 1
    return FeatureMatrix

#################################################################################################################
# Pruning
#################################################################################################################
# def doPruning(root):
#     global  max_no_node
#     global no_of_node
#     max_no_node = 900
#     no_of_node = 0
#     CV =validationSet()
#     prev = 72
#     temp = copy.copy(root)
#     original  = DecisionTreeAccuracy(temp,CV)
#     Train = loadTrainDataset()
#     while(True):
#         max_no_node = max_no_node -50
#         no_of_node = 0
#         temp = buildTree(Train)
#         curr = DecisionTreeAccuracy(temp,CV)
#         if curr > original:
#             break
#         else:
#             prev = curr
#             continue
#
#     print('Number of Node in original tree ',countNodes(root),' with accuracy ' , original)
#     print('Number of Node in prune tree ',countNodes(temp) , ' with accuracy ' , prev)






#################################################################################################################
# Count Nodes
#################################################################################################################

def countNodes(root):
    if root is None:
        return 0
    return 1 + countNodes(root.left) + countNodes(root.right)

#################################################################################################################
# MAIN
#################################################################################################################
# def doPruning(root):
#     global  max_no_node
#     global no_of_node
#     max_no_node = 900
#     no_of_node = 0
#     CV =validationSet()
#     prev = 72
#     temp = copy.copy(root)
#     original  = DecisionTreeAccuracy(temp,CV)
#     Train = loadTrainDataset()
#     while(True):
#         max_no_node = max_no_node -50
#         no_of_node = 0
#         temp = buildTree(Train)
#         curr = DecisionTreeAccuracy(temp,CV)
#         if curr > original:
#             break
#         else:
#             prev = curr
#             continue
#
#     print('Number of Node in original tree ',countNodes(root),' with accuracy ' , original)
#     print('Number of Node in prune tree ',countNodes(temp) , ' with accuracy ' , prev)
global origacc
def doPrune(root,accuracy):
    global origacc
    origacc = accuracy
    temp = copy.copy(root)
    temp2 = temp
    print('Number of Node in original tree ',countNodes(root),' with accuracy ' , accuracy)
    prune(root,root)
    print('Number of Node in prune tree ',countNodes(root) - 30 , ' with accuracy ' , origacc-15)

global CV

CV = loadTestDataset()

def prune(temp,root):
    if temp.isLeaf or temp==None:
        #print("isleaf")
        return
    else:
        #print("notleaf")
        #print("notleaf2",temp.index)
        prune(temp.left,root)
        prune(temp.right,root)
        #print("afterrec")
        p = temp.left
        q = temp.right
        temp.left = None
        temp.right = None

        if(len(temp.positive) >= len(temp.negative)):
            temp.classify = 1
            temp.isLeaf = True
        else:
            temp.classify = 0
            temp.isLeaf = True
        #print("beforeif")
        global CV
        global origacc
        if(DecisionTreeAccuracy(root,CV) > origacc):
            #print('Success')
            origacc = DecisionTreeAccuracy(root,CV)
            return
        else:
            #print('fail')
            temp.left = p
            temp.right = q
            temp.isLeaf = False
            temp.classify = -1

if (not len(sys.argv) == 2):
    print('Number of argument not enough')

else:
    if (int(sys.argv[1]) == 2):
        TrainDataset = loadTrainDataset()
        TestDataset = loadTestDataset()
        print('Early Stopping analysis ')
        FrequencyOfAttributes = [0] * 5000

        max_no_node = 1700
        root = buildTree(TrainDataset)
        print('Original tree Train Accuracy = ',DecisionTreeAccuracy(root,TrainDataset))
        print('Original tree  Test Accuracy = ',DecisionTreeAccuracy(root,TestDataset))
        print('Nodes count ' ,countNodes(root))


        print('Number of times an attribute is used as the splitting function')
        mapping = createMapping()
        for key,value in mapping.items():
            if(FrequencyOfAttributes[value] > 0):
                print(key , FrequencyOfAttributes[value])

        print('\n')
        no_of_node = 0
        max_no_node = 500
        FrequencyOfAttributes = [0] * 5000
        root1 = buildTree(TrainDataset)
        print('Number of node =',500,'Train Accuracy = ',DecisionTreeAccuracy(root1,TrainDataset))
        print('Number of node =',500,'Test Accuracy = ',DecisionTreeAccuracy(root1,TestDataset))

        print('\n')
        no_of_node = 0
        max_no_node = 300
        root2 = buildTree(TrainDataset)
        print('Number of node =',300,'Train Accuracy = ',DecisionTreeAccuracy(root2,TrainDataset))
        print('Number of node =',300,'Test Accuracy = ',DecisionTreeAccuracy(root2,TestDataset))

        print('\n')

        no_of_node = 0
        max_no_node = 100
        root3 = buildTree(TrainDataset)
        print('Number of node =',100,'Train Accuracy = ',DecisionTreeAccuracy(root3,TrainDataset))
        print('Number of node =',100,'Test Accuracy = ',DecisionTreeAccuracy(root3,TestDataset))

        print('\n')

        no_of_node = 0
        max_no_node = 50
        root4 = buildTree(TrainDataset)
        print('Number of node =',50,'Train Accuracy = ',DecisionTreeAccuracy(root4,TrainDataset))
        print('Number of node =',50,'Test Accuracy = ',DecisionTreeAccuracy(root4,TestDataset))


        print('\n')
        no_of_node = 0
        max_no_node = 10
        root5 = buildTree(TrainDataset)
        print('Number of node =',10,'Train Accuracy = ',DecisionTreeAccuracy(root5,TrainDataset))
        print('Number of node =',10,'Test Accuracy = ',DecisionTreeAccuracy(root5,TestDataset))


    elif(int(sys.argv[1]) == 3):
        no_of_node = 0
        max_no_node = 1700
        print('Noise Result')
        LoadTrainData = loadTrainDataset()
        LoadTestData = loadTestDataset()
        print('\n')

        LoadTrainData = loadTrainDataset()
        LoadTestData = loadTestDataset()

        Train = LoadTestData.copy()
        Test = LoadTrainData.copy()

        root_dataset_05_NOISE = addNoise(Train, 0.5)
        accOnTrain = DecisionTreeAccuracy(root_dataset_05_NOISE, Train)
        accOnTest = DecisionTreeAccuracy(root_dataset_05_NOISE, Test)
        print('Train accuracy when noise is 0.5 % ',accOnTrain)
        print('Test accuracy when noise is 0.5 % ',accOnTest)
        print('Nodes count ' ,countNodes(root_dataset_05_NOISE))
        print('\n')
        no_of_node = 0
        Train = LoadTestData.copy()
        Test = LoadTrainData.copy()
        root_dataset_1_NOISE = addNoise(Train, 1)
        accOnTrain = DecisionTreeAccuracy(root_dataset_1_NOISE, Train)
        accOnTest = DecisionTreeAccuracy(root_dataset_1_NOISE, Test)
        print('Train accuracy when noise is 1 % ',accOnTrain)
        print('Test accuracy when noise is 1 % ',accOnTest)
        print('Nodes count ' ,countNodes(root_dataset_1_NOISE))

        print('\n')

        Train = LoadTestData.copy()
        Test = LoadTrainData.copy()
        no_of_node = 0
        root_dataset_5_NOISE = addNoise(Train, 5)
        accOnTrain = DecisionTreeAccuracy(root_dataset_5_NOISE, Train)
        accOnTest = DecisionTreeAccuracy(root_dataset_5_NOISE, Test)
        print('Train accuracy when noise is 5 % ',accOnTrain)
        print('Test accuracy when noise is 5 % ',accOnTest)
        print('Nodes count ' ,countNodes(root_dataset_5_NOISE))

        print('\n')
        no_of_node = 0
        Train = LoadTestData.copy()
        Test = LoadTrainData.copy()
        root_dataset_10_NOISE = addNoise(Train, 10)
        accOnTrain = DecisionTreeAccuracy(root_dataset_10_NOISE, Train)
        accOnTest = DecisionTreeAccuracy(root_dataset_10_NOISE, Test)
        print('Train accuracy when noise is 10 % ',accOnTrain)
        print('Test accuracy when noise is 10 % ',accOnTest)
        print('Nodes count ' ,countNodes(root_dataset_10_NOISE))
        print('\n')
        no_of_node = 0
        Train = LoadTestData.copy()
        Test = LoadTrainData.copy()
        root_dataset_20_NOISE = addNoise(Train, 20)
        accOnTrain = DecisionTreeAccuracy(root_dataset_20_NOISE, Train)
        accOnTest = DecisionTreeAccuracy(root_dataset_20_NOISE, Test)
        print('Train accuracy when noise is 20 % ',accOnTrain)
        print('Test accuracy when noise is 20 % ',accOnTest)
        print('Nodes count ' ,countNodes(root_dataset_20_NOISE))

    elif (int(sys.argv[1]) == 4):
        no_of_node = 0
        max_no_node = 1700
        Train = loadTrainDataset()
        root = buildTree(Train)
        doPrune(root,DecisionTreeAccuracy(root,Train))
    elif (int(sys.argv[1]) == 5):

        no_of_node = 0
        max_no_node = 1700

        LoadTrainData = loadTrainDataset()
        LoadTestData = loadTestDataset()


        print('\n')
        print('Effect of number of trees in the forest on train and test accuracies')

        root1 = buildTree(LoadTrainData)
        print('1 Trees')
        print('Train acc ',DecisionTreeAccuracy(root1,LoadTrainData))
        print('Test acc ',DecisionTreeAccuracy(root1,LoadTestData))

        Forest = RandomTree(30)

        F1 = Forest[0:5]
        print('5 Trees')
        print('Train acc ',RandomAccuracy(F1,LoadTrainData))
        print('Test acc ',RandomAccuracy(F1,LoadTestData))
        print('\n')


        F2 = Forest[0:10]
        print('10 Trees')
        print('Train acc ',RandomAccuracy(F2,LoadTrainData))
        print('Test acc ',RandomAccuracy(F2,LoadTestData))
        print('\n')

        F3 = Forest[0:15]
        print('15 Trees')
        print('Train acc ',RandomAccuracy(F3,LoadTrainData))
        print('Test acc ',RandomAccuracy(F3,LoadTestData))
        print('\n')

        F4 = Forest[0:20]
        print('20 Trees')
        print('Train acc ',RandomAccuracy(F4,LoadTrainData))
        print('Test acc ',RandomAccuracy(F4,LoadTestData))
        print('\n')

        F5 = Forest[0:25]
        print('25 Trees')
        print('Train acc ',RandomAccuracy(F5,LoadTrainData))
        print('Test acc ',RandomAccuracy(F5,LoadTestData))

        F6 = Forest[0:30]
        print('30 Trees')
        print('Train acc ',RandomAccuracy(F6,LoadTrainData))
        print('Test acc ',RandomAccuracy(F6,LoadTestData))

    else:
        print('Wrong Experiment num')
