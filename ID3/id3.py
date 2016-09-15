import math
import pickle

def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'no'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'filppers']
    return dataset, labels


def calEntropy(dataset):
    """计算数据集的熵值

    :param dataset: 数据集
    :return: 熵值
    """
    entropy = 0
    numberOfDataSet = len(dataset)
    # 计算所有标签的个数
    labelcount = {}
    for data in dataset:
        labelcount[data[-1]] = labelcount.get(data[-1], 0) + 1
    # 计算熵值
    for value in labelcount.values():
        prob = value / numberOfDataSet
        entropy -= prob * math.log(prob, 2)
    return entropy


def splitDataSet(dataset, featureindex, value):
    """获取特征值等于的value的数据集

    :param dataset: 数据集
    :param featureindex: 特征列索引
    :param value: 特征值
    :return: 新数据集
    """
    newDataset = []
    # 如果数据的特征值等于value则将其添加到新数据集，并删除该特征列
    for data in dataset:
        if data[featureindex] == value:
            newDataset.append(data[:featureindex] + data[featureindex + 1:])
    return newDataset


def chooseBestFeature(dataset):
    """从当前的数据集中选择最大信息增益的特征分类

    :param dataset: 数据集
    :return: 特征索引
    """
    baseEntropy = calEntropy(dataset)
    classEntropy = 0
    bestFeature = -1
    bestRange = -1
    numberOfFeatures =  len(dataset[0]) - 1
    # 计算每个特征划分的信息熵
    for featureindex in range(numberOfFeatures):
        uniquefeatureValue = set([value for value in dataset[featureindex]])
        for value in uniquefeatureValue:
            newDataSet = splitDataSet(dataset, featureindex, value)
            entropy = calEntropy(newDataSet)
            classEntropy += len(newDataSet) / len(dataset) * entropy
        if baseEntropy - classEntropy > bestRange:
            bestRange = baseEntropy - classEntropy
            bestFeature = featureindex
    return bestFeature


def buildDecisionTree(dataset, labels):
    """构建决策树

    算法思路：
        1. 寻找划分数据集的最好特征（信息增益最多的特征）
        2. 根据特征值的所有情况划分数据集
        3. 递归地将子数据集建树（回到第一步）
    格式：
        {'no surfacing': {0: 'no', 1: {'filppers': {0: 'no', 1: 'yes'}}}}
    :param dataset: 数据集（最后一列是分类）
    :param labels: 标签（每一层结点的名字，比如 ‘男性’，‘30岁’）
    :return: 决策树
    """
    # 当所有数据分类一致时返回分类
    if dataset.count([classifydata for classifydata in dataset[-1]]) == len(dataset):
        return dataset[0][-1]
    # 当特征数用尽但分类不纯时，进行多数表决返回分类
    if len(dataset[0]) == 1:
        return majorityVote(dataset)
    # 选择当前信息增益最多的特征
    featureIndex = chooseBestFeature(dataset)
    featureValue = [data[featureIndex] for data in dataset]
    uniqueFeatureValue = set(featureValue)
    tree = {labels[featureIndex]: {}}
    # 对该特征的所有值创建结点并递归建树
    for value in uniqueFeatureValue:
        # 因为del会影响到原标签，所以copy一个对象
        newLabels = labels[:]
        del(newLabels[featureIndex])
        # 每一种特征值进行一次划分数据集
        tree[labels[featureIndex]][value] = buildDecisionTree(splitDataSet(dataset, featureIndex, value), newLabels)
    return tree


def majorityVote(dataset):
    """多数表决

    当特征数已经用尽，但是分类不纯时选择分类数据个数较多分类
    :param dataset: 数据集
    :return:
    """
    labelcount = {}
    for data in dataset:
        labelcount[data[-1]] = labelcount.get(data[-1], 0) + 2
    sortlabelcount = sorted(labelcount.items(), key=lambda i: i[1], reverse=True)
    return sortlabelcount[0][0]


def getTreeHeight(tree):
    """获取决策树高度

    :param tree: 决策树
    :return: 树高
    """
    treeHeight = 0
    # 获取标签关键词（为了去除标签层，不然层数会多加上标签层）
    label = list(tree.keys())[0]
    for value in tree[label].values():
        if type(value).__name__ == 'dict':
            currentTreeHeight = 1 + getTreeHeight(value)
        else:
            currentTreeHeight = 1
        # 如果有多个结点（字典），取层数最高的作为树高
        if currentTreeHeight > treeHeight:
            treeHeight = currentTreeHeight
    return treeHeight


def getLeafNumber(tree):
    """获取决策树的叶子结点个数（也就是分类）

    :param tree: 决策树
    :return: 叶子结点个数
    """
    leafNumber = 0
    label = list(tree.keys())[0]
    for value in tree[label].values():
        if type(value).__name__ == 'dict':
            leafNumber += getLeafNumber(value)
        else:
            leafNumber += 1
    return leafNumber


def classifyPredict(tree, featurelabels, featureValuelist):
    """对新数据进行分类

    由于决策树的每一层结点标签不一定对应新数据的值，所以需要先找到决策树标签相应的数据值再匹配判断
    :param tree: 决策树
    :param featurelabels: 新数据的标签（列表）
    :param featureValuelist: 新数据的值（列表）
    :return: 分类结果
    """
    label = list(tree.keys())[0]
    # 找到新数据在该层标签的值
    featureIndex = featurelabels.index(label)
    featureValue = featureValuelist[featureIndex]
    for key, value in tree[label].items():
        # 如果新数据的值和某一种情况匹配
        if key == featureValue:
            if type(value).__name__ == 'dict':
                predict = classifyPredict(value, featurelabels, featureValuelist)
            else:
                predict = value
    return predict


def saveTree(tree, filename):
    """序列化操作，保存决策树

    :param tree: 决策树
    :param filename: 保存文件名
    :return:
    """
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def restoreTree(filename):
    """反序列化操作，重新加载上一次保存数据

    :param filename: 文件名
    :return: 数据
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    dataset, labels = createDataSet()
    # print(dataset)
    # print(calEntropy(dataset))
    # print(splitDataSet(dataset, 0, 1))
    # print(chooseBestFeature(dataset))
    tree = buildDecisionTree(dataset, labels)
    # print(tree)
    # print(getTreeHeight(tree))
    # print(getLeafNumber(tree))
    # print(classifyPredict(tree, ['filppers', 'no surfacing'], [0, 1]))
    print(saveTree(tree, 'myDecisionTree.txt'))
    print(restoreTree('myDecisionTree.txt'))