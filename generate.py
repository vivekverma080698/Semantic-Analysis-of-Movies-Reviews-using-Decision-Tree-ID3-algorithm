import random
from random import randint


def PrintToFile(data, a):
    if (a == 1):
        with open('MyTrainData.txt', 'w') as f:
            for key, value in data:
                f.write('%s\n' % (value))

    if (a == 2):
        with open('MyTestData.txt', 'w') as f:
            for key, value in data:
                f.write('%s\n' % (value))

    if (a == 3):
        with open('MyVocab.txt', 'w') as f:
            for value in data:
                f.write('%s\n' % (value))


def getTrainDataset():
    No_of_Instance = 1000
    No_of_feature = 5000

    with open('imdbEr.txt', 'r') as myfile:
        FileExp = myfile.read().splitlines()

    with open('imdb.vocab', 'r') as myfile:
        FileVocab = myfile.read().splitlines()


    Worddict = {}
    for i in range(len(FileExp)):
        Worddict[FileVocab[i]] = FileExp[i]

    NewPve = {}  # selected positive words
    NewNeg = {}  # selected negative words
    WordIndex = {}
    PosIndex = {}  # +ve word with index in vocablury
    NegIndex = {}  # -ve word with index in vocablury
    i = 0;
    for key, value in Worddict.items():
        if (float(value) > 2.2):
            NewPve[key] = value
            PosIndex[key] = i
        if (float(value) < -1.2):
            NewNeg[key] = value
            NegIndex[key] = i
        WordIndex[key] = i
        i = i + 1

    PRS = random.sample(PosIndex.items(), int(No_of_feature/2))
    PSS = random.sample(NegIndex.items(), int(No_of_feature/2))

    MapIndexToFeat = {}
    FeatureMapIndex = []
    index = 0
    for key, value in PRS:
        MapIndexToFeat[value] = index
        FeatureMapIndex.append(value)
        index += 1
    for key, value in PSS:
        if (index < 5000):
            MapIndexToFeat[value] = index
            FeatureMapIndex.append(value)
            index += 1

    PrintToFile(FeatureMapIndex, 3)

    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('train/labeledBow.feat', 'r') as myfile:
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

    TDP = random.sample(TrainDatasetP.items(), int(No_of_Instance / 2))
    TDN = random.sample(TrainDatasetN.items(), int(No_of_Instance / 2))

    Training = TDP + TDN

    PrintToFile(Training, 1)


def getTestDataset():
    No_of_Instance = 1000
    TrainDatasetP = {}
    TrainDatasetN = {}
    with open('test/labeledBow.feat', 'r') as myfile:
        line = myfile.readline().strip();
        cnt = 0
        noofpf = 1;
        noofnf = 1;
        while (line):
            if ((int(line[0:2]) >= 7)):
                TrainDatasetP[cnt] = line;
                noofpf = noofpf + 1;
            if (int(line[0:2]) <= 4):
                TrainDatasetN[cnt] = line;
                noofnf = noofnf + 1;
            cnt += 1
            line = myfile.readline().strip();

    TDP = random.sample(TrainDatasetP.items(), int(No_of_Instance / 2))
    TDN = random.sample(TrainDatasetN.items(), int(No_of_Instance / 2))

    Training = TDP + TDN
    PrintToFile(Training, 2)

getTrainDataset()
getTestDataset()
print('Your Data Is Generated')