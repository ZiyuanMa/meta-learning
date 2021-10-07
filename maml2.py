import os
from typing import List
import torch
import numpy as np

# root_path = './'
# processed_folder =  os.path.join(root_path)

# zip_ref = zipfile.ZipFile(os.path.join(root_path,'images_background.zip'), 'r')
# zip_ref.extractall(root_path)
# zip_ref.close()
# zip_ref = zipfile.ZipFile(os.path.join(root_path,'images_evaluation.zip'), 'r')
# zip_ref.extractall(root_path)
# zip_ref.close()
# root_dir = './'

# import zipfile
root_path = './'
file_names = ['images_background.zip', 'images_evaluation.zip']
# for file_name in file_names:
#     with zipfile.ZipFile(root_path+file_name, "r") as zip_ref:
#         zip_ref.extractall(root_path)



# # 数据预处理
import torchvision.transforms as transforms
from PIL import Image

'''
an example of img_items:
( '0709_17.png',
  'Alphabet_of_the_Magi/character01',
  './../datasets/omniglot/python/images_background/Alphabet_of_the_Magi/character01')
'''

root_dir_train = os.path.join(root_path, 'images_background')
root_dir_test = os.path.join(root_path, 'images_evaluation')

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                lambda img: img / 255,
                                ])

data_set = []
for (root, dirs, files) in os.walk(root_dir_train):
    # print(f'{root}, {dirs}, {files}')
    if files:
        data = []
        for file in files:
            image = Image.open(os.path.join(root, file))
            # print(image.shape)
            image = transform(image)
            data.append(image)

        assert len(data) == 20, f'only {len(data)} data'
        data_set.append(data)
    
print(f'num of class: {len(data_set)}')
# def find_classes(root_dir_train):
#     img_items = []
#     for (root, dirs, files) in os.walk(root_dir_train): 
#         for file in files:
#             if (file.endswith("png")):
#                 r = root.split('/')
#                 img_items.append((file, r[-2] + "/" + r[-1], root))
#     print("== Found %d items " % len(img_items))
#     return img_items


### 准备数据迭代器
num_ways = 5
support = 1  ## support data 的个数
query = 15 ## query data 的个数
num_tasks = 32

from typing import List
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, dataloader

class NewDataset(Dataset):
    '''初始化数据集'''
    def __init__(self, data_set: List[List[torch.Tensor]], num_tasks, num_ways, query, support):
        self.data_set = data_set
        self.num_classes = len(data_set)
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = support
        self.support = support
        self.query = query


    '''根据下标返回数据(img和label)'''
    def __getitem__(self, index):
        
        supports, querys = [], []
        
        for i in range(index, index+self.num_ways):
            i = i % self.num_classes
            cases = random.sample(self.data_set[i], self.support+self.query)
            random.shuffle(cases)
            supports.extend(cases[:self.support])
            querys.extend(cases[self.support:])
        
        return torch.stack(supports), torch.stack(querys)

    '''返回数据集长度'''
    def __len__(self):
        return len(self.data_set)





import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy, copy

class Network(nn.Module):
    def __init__(self):
        super().__init__()


        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64, num_ways),
        )


        
    def forward(self, x):
        '''
        :bn_training: set False to not update
        :return: 
        '''

        
        return self.network(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Network()
net = net.to(device)
net.train()
optim = torch.optim.SGD(net.parameters(), lr=0.1)

data_set = NewDataset(data_set, num_tasks, num_ways, query, support)
data_loader = dataloader.DataLoader(data_set, batch_size=num_tasks, shuffle=True)

support_target = torch.arange(num_ways).repeat_interleave(support)
query_target = torch.arange(num_ways).repeat_interleave(query)

support_target = support_target.to(device)
query_target = query_target.to(device)

sec_order = False

for _ in range(5):
    epoch_loss = 0
    for supports, querys in data_loader:
        querys, supports = querys.to(device), supports.to(device)

        total_grads = []

        net.zero_grad()

        for support, query in zip(supports, querys):
            task_net = deepcopy(net)
            task_optim = torch.optim.SGD(task_net.parameters(), lr=0.001)

            support_out = task_net(support)
            # print(support_out.size())
            # print(support_target.size())
            loss = F.cross_entropy(support_out, support_target)
            loss.backward(retain_graph=sec_order, create_graph=sec_order)
            task_optim.step()
            task_net.zero_grad()

            query_out = task_net(query)
            loss = F.cross_entropy(query_out, query_target)
            epoch_loss += loss.item()
            task_grads = torch.autograd.grad(loss, task_net.parameters())

            total_grads.append(task_grads)

        avg_grads = [sum(total_grad) / num_tasks for total_grad in zip(*total_grads)]
        
        for param, avg_grad in zip(net.parameters(), avg_grads):
            param.grad = avg_grad

        optim.step()
    
    print(epoch_loss)
        

    #     print(querys.size())
        # for query, query_target, support, support_target in zip(querys, querys_target, supports, supports_target):
        #     print(query.size())


## omniglot
import random
random.seed(1337)
np.random.seed(1337)

import time
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

meta = MetaLearner().to(device)

epochs = 60001
for step in range(epochs):
    start = time.time()
    x_spt, y_spt, x_qry, y_qry = next('train')
    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device),\
                                 torch.from_numpy(y_spt).to(device),\
                                 torch.from_numpy(x_qry).to(device),\
                                 torch.from_numpy(y_qry).to(device)
    accs,loss = meta(x_spt, y_spt, x_qry, y_qry)
    end = time.time()
    if step % 100 == 0:
        print("epoch:" ,step)
        print(accs)
#         print(loss)
        
    if step % 1000 == 0:
        accs = []
        for _ in range(1000//task_num):
            # db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device),\
                                         torch.from_numpy(y_spt).to(device),\
                                         torch.from_numpy(x_qry).to(device),\
                                         torch.from_numpy(y_qry).to(device)

            
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append(test_acc)
        print('在mean process之前：',np.array(accs).shape)
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('测试集准确率:',accs)

