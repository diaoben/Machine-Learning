import math


def calculateInformationentropy(data):
    #计算信息熵
    entropy = 0
    datas={};
    count = len(data)
    for element in data:
        datas[element[-1]] = datas.get(element[-1],0)+1
    for v in datas:
        entropy -= v/count * math.log(v/count,2);
    return entropy;

def getSubDatasAtValue(datas,index,value):
    newdata = [];
    for data in datas:
        if data[index] == value:
            newdata.append(datas[:index]+datas[index+1:])
    return newdata;


def getBestAttribute(datas):
    #获得最大相对熵下标
    entropy = calculateInformationentropy(datas)
    bestindex = -1;
    bestRate = -1;
    nrOfAttribute = len(datas[0]) - 1;
    for index in range(nrOfAttribute):
        attributes = set(data[index] for data in datas)
        subentropy = 0;
        for attr in attributes:
            subdata = getSubDatasAtValue(datas,index,attr)
            subentropy += len(subdata) /len(datas) * calculateInformationentropy(subdata);
        if entropy - subentropy > bestRate:
            bestindex = index;
            bestRate = entropy - subentropy;

    return bestindex;

def generateTree(datas,labels):
    if datas.count(datas[-1])==len(datas):
        return datas[0][-1]
    if len(datas[0]) == 1:
        counts = {};
        results = set(data[-1] for data in datas);
        for r in results:
            counts[r] = datas.count(r);
        sortlabelcount = sorted(results.items(), key=lambda i: i[1], reverse=True)
        return sortlabelcount[0][0]

    bestindex = getBestAttribute(datas);
    tree = {labels[bestindex]:{}}
    newlabels = labels[:]
    del(newlabels[bestindex])

    attributes = set(data[bestindex] for data in datas)
    for attr in attributes:
        tree[labels[bestindex]][attr] = generateTree(getSubDatasAtValue(datas,bestindex,attr),newlabels)
    return tree;



def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'no'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'filppers']
    return dataset, labels

if __name__ == '__main__':
    dataset, labels = createDataSet()
    # print(dataset)
    # print(calEntropy(dataset))
    # print(splitDataSet(dataset, 0, 1))
    # print(chooseBestFeature(dataset))
    tree = generateTree(dataset, labels)
    print(tree)



