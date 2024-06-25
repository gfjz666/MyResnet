import numpy as np
import scipy.io as scio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import Mat_deal

# 读取mat中频数据
traindata = Mat_deal.readmat('./input_data/s512_10db.mat')
trainlabel = Mat_deal.label_gen(0)

# 训练测试集分割
traindata, testdata, trainlabel, testlabel = train_test_split(traindata, trainlabel, test_size=0.3, random_state=20,
                                                              shuffle=True, stratify=trainlabel)
# 测试验证集分割
testdata, valdata, testlabel, vallabel = train_test_split(testdata, testlabel, test_size=1 / 3, random_state=20,
                                                          shuffle=True, stratify=testlabel)

# 重写dataset class
class MyDataset(Dataset):

    def __init__(self, data, label, type):
        # fh = open(txt_path, 'r')
        # 输入numpy中频data
        # fh=csv.reader(open(txt_path))
        self.type = type
        row = data.shape[0]  # 行数

        Ifdata = []  # 中频和标签集合
        for i in range(0, row):
            Ifdata.append((data[i, :], label[i]))
            self.Ifdata = Ifdata

    def __getitem__(self, index):
        data, label = self.Ifdata[index]

        # data = data[500:1500]

        # 输入IQ两路信号
        data = np.vstack((data.real, data.imag))

        data = torch.from_numpy(data).to(torch.float32)
        data = data.unsqueeze(0)
        data = data.transpose(1, 2)

        # 张量转换
        label = torch.from_numpy(label)
        label = label.squeeze(-1)

        return data, label

    def __len__(self):
        return len(self.Ifdata)


train_data = MyDataset(traindata, trainlabel, 'train')
test_data = MyDataset(testdata, testlabel, 'test')
val_data = MyDataset(valdata, vallabel, 'val')
# print(float(len(train_data)))

batch_size = 128

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# for inputs, labels in test_loader:
#     print(labels)
# x, y = next(iter(test_loader))
# print("x.size: ", x.size())
# print("y.size: ", y.size())
