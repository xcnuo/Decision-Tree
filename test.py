import C45
import treePlotter
fr = open(r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\Data.txt')
lDataSet = [inst.strip().split('\t') for inst in fr.readlines()]
labels = ['Job', 'Age', 'Credit score', 'House']
# 样本特征类型，0为离散，1为连续
labelProperties = [0, 1, 1, 0]
# 是否放贷
classList = ['Yes', 'No']
# 验证集，用于剪枝
dataSet_test = [['full-time job', '35', '85', 'have house', 1, 'Yes'], ['no job', '72', '90', 'have house', 1, 'No']]
# 构建决策树
trees = C45.createTree(lDataSet, labels, labelProperties)
treePlotter.createPlot(trees)
# 利用验证集对决策树剪枝
C45.postPruningTree(trees, classList, lDataSet, dataSet_test, labels, labelProperties)
# 绘制剪枝后的决策树
treePlotter.createPlot(trees)
# 重新赋值类别标签和类型
labels = ['Job', 'Age', 'Credit score', 'House']
labelProperties = [0, 1, 1, 0]
# 测试样本
testVec = ['part-time job', 50, 88, 'have house']
classLabel = C45.classify(trees, classList, labels, labelProperties, testVec)
# 打印测试样本的分类结果
print(classLabel)


