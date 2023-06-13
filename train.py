import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import datasets, utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os




# 定义resnet50使用的残差块Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 替换输入数据，节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        # 定义shortcut部分
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 定义网络结构
class Resnet50(nn.Module):
    def __init__(self, Bottleneck, num_class=10):
        super(Resnet50, self).__init__()
        self.in_channels = 64

        # 第一层没有残差块，单独设计
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 第一个卷基层输入为rgb，维数默认为3
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)

    # 定义每个重复块的方法
    def _make_layer(self, block, out_channels, stride, padding):
        layer = []
        flag = True
        for i in range(0, len(stride)):
            layer.append(block(self.in_channels, out_channels, stride=stride[i], padding=padding[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# 数据预处理，通过transforms处理后输入的大小为224*224*3，每个像素的值归一化到[-1,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    transforms.Resize((224, 224))
])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

testing_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# 定义网络
res50 = Resnet50(Bottleneck)

# 将模型加载到gpu上
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = res50.to(device)

# 设置学习率
learning_rate = 1e-4

# 设置损失函数，由于是多分类(>2)选择交叉熵损失
loss_fn = nn.CrossEntropyLoss()

# 设置优化器，防止局部最优解，同时实现自适应学习率，选择adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 加载训练集和数据集
batch_size = 100
train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=True)

# 开始训练
epoch = 5
for i in range(epoch):
    print("Epoch {}/{}".format(i + 1, epoch))
    print("-" * 10)
    running_loss = 0.0
    i=0
    for image, label in train_data:
        i+=1
        image = image.cuda()
        label = label.cuda()
        # 训练模型得到预测值
        label_pred = model(image)
        # 计算损失
        loss = loss_fn(label_pred, label)
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 根据梯度更新权重
        optimizer.step()
        running_loss += loss.item()
        print("has done {}/1000 batch and loss is {}".format(i,running_loss))

file_name = 'cifar10_resnet.pt'
torch.save(model, file_name)
print(file_name+' saved successfully!')

# 测试
model = torch.load('cifar10_resnet.pt') #加载模型
model.eval() #切换到测试模式

correct = 0
total = 0
i=0
for image, labels in test_data:
    i+=1
    image = image.cuda()
    labels = labels.cuda()
    # 前向传播
    with torch.no_grad():
        out = model(image)
    # 求出预测值索引
    _, predicted = torch.max(out.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print("has done {}/200 batch and accuracy is {}".format(i, correct))
print('10000测试图像 准确率:{:.4f}%'.format(100 * correct / total))