import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def set_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，还需要设置 CUDA 的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 PyTorch 的确定性行为（可选，但有助于可重复性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置一个固定的种子值，例如 42
set_seed(42)


class BCdataset(Dataset):

    def __init__(self, in_file, out_file, transform=None):
        self.input_frame = in_file
        self.label_frame = out_file
        self.transform = transform

    def __len__(self):
        return len(self.input_frame)

    def __getitem__(self, idx):
        input = self.input_frame[idx, :]
        label = self.label_frame[idx, :]
        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label).float()
        return input, label


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2,n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3,n_output)
        self._initialize_weights()

    def _initialize_weights(self):
        # 对每一层进行 Xavier 初始化
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.zeros_(self.hidden1.bias)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        torch.nn.init.zeros_(self.hidden2.bias)
        torch.nn.init.xavier_uniform_(self.hidden3.weight)
        torch.nn.init.zeros_(self.hidden3.bias)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        torch.nn.init.zeros_(self.predict.bias)
        # torch.nn.init.kaiming_uniform_(self.hidden1.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.hidden1.bias)

    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        y = self.predict(x)
        return y


def NN_train(train,test,lr,num_iterations):
    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    base_path       = './False/Re100_100episode_False_42'
    Path_to_weights = os.path.join(base_path, 'weight_storage')
    image_path      = os.path.join(base_path, 'image')
    os.makedirs(Path_to_weights, exist_ok=True)
    os.makedirs(image_path,      exist_ok=True)

    bs, bst         = 8000,8000
    train_data      = DataLoader(train, shuffle=True,  batch_size=bs)
    test_data       = DataLoader(test,  shuffle=False, batch_size=bst)
    dataloaders     = {'train': train_data, 'test': test_data}
    dataset_sizes   = {'train': len(train_data.dataset), 'test': len(test_data.dataset)}

    model     = Net(152, 304, 304, 304, 152).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_loss_list = []
    test_loss_list  = []
    best_test_loss  = float('inf')  # 初始化为正无穷
    best_model_wts  = None

    for epoch in range(num_iterations):
        # 训练阶段
        model.train()
        train_loss_sum = 0.0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * inputs.size(0)  # 乘以batch样本数

        # 测试阶段
        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss_sum += loss.item() * inputs.size(0)

        # 计算平均损失
        epoch_train_loss = train_loss_sum / dataset_sizes['train']
        epoch_test_loss = test_loss_sum / dataset_sizes['test']
        train_loss_list.append(epoch_train_loss)
        test_loss_list.append(epoch_test_loss)

        # 学习率调整
        scheduler.step(epoch_test_loss)

        # 保存最佳模型
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(Path_to_weights, f'NN_weights_best.pth'))

        # 定期保存模型和打印信息
        if epoch % 1000 == 0 or epoch == num_iterations - 1:
            print(f'Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(Path_to_weights, f'NN_weights_step{epoch}.pth'))

    # 保存损失曲线
    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(image_path, 'InputToLabel_loss_curve.png'))
    plt.close()

        # 新增代码：保存损失数据到文本文件
    with open(os.path.join(image_path, 'InputToLabel_train_loss.txt'), 'w') as f:
        for loss in train_loss_list:
            f.write(f"{loss}\n")  # 每个损失值占一行

    with open(os.path.join(image_path, 'InputToLabel_test_loss.txt'), 'w') as f:
        for loss in test_loss_list:
            f.write(f"{loss}\n")


    # 保存最佳模型权重
    torch.save(best_model_wts, os.path.join(Path_to_weights, 'NN_weights.pth'))
    print(f"Best Test Loss: {best_test_loss:.4f}")


def NN_pred(x_pred):
    Path_to_weights = './data/Re1000_100episode/weight_storage'
    weights_name = "NN_weights.pth"
    cd_weights = os.path.join(Path_to_weights, weights_name)
    model = Net(152, 304, 304, 304, 152)
    model.load_state_dict(torch.load(cd_weights))
    x_pred = torch.tensor(x_pred).to(torch.float32)
    y = model(x_pred)
    y = y.detach().numpy()
    return y


def read_data():
    folder_path = './Re1000data'
    number_practice = 100

    action_path = './Re1000data/99/action.csv'
    action_pd = pd.read_csv(action_path, header=None)
    action_arr = action_pd.values
    # action_arr = action_arr.reshape(-1,80)
    # action_arr = action_arr[:, 2:].reshape(-1, 1)

    cd_list = []
    cl_list = []
    for i in range(number_practice):
        cd_path = os.path.join(folder_path, str(i), 'cdcl.csv')
        df = pd.read_csv(cd_path, skiprows=2, sep=' ')
        for j in range(0, 4000, 50):
            a = df.iloc[j:j + 50, 1]
            b = df.iloc[j:j + 50, 2]
            a = a.mean()
            b = b.mean()
            b = np.abs(b)
            cd_list.append(a)
            cl_list.append(b)

    cd_arr = np.array(cd_list).reshape(-1, 1)
    cl_arr = np.array(cl_list).reshape(-1, 1)
    # cd_arr = np.array(cd_list).reshape(-1, 80)
    # cl_arr = np.array(cl_list).reshape(-1, 80)
    # cd_arr = cd_arr[:, 2:].reshape(-1, 1)
    # cl_arr = cl_arr[:, 2:].reshape(-1, 1)



    index = 0
    # state_arr = np.zeros((number_practice * 78, 151))
    # next_state_arr = np.zeros((number_practice * 78, 151))
    state_arr = np.zeros((number_practice * 80, 151))
    next_state_arr = np.zeros((number_practice * 80, 151))

    for i in range(number_practice):
        for j in range(80):
            path = os.path.join(folder_path, str(i), 'flowfield', 'episode_' + str(j) + '.csv')
            next_path = os.path.join(folder_path, str(i), 'flowfield', 'episode_' + str(j + 1) + '.csv')
            df = pd.read_csv(path, usecols=[3])
            next_df = pd.read_csv(next_path, usecols=[3])
            data_array = df.values
            next_data_array = next_df.values
            state_arr[index, :] = data_array.flatten()
            next_state_arr[index, :] = next_data_array.flatten()
            index += 1


    reward_arr = (-1)*cd_arr +(-0.2)*cl_arr +3.13

    inputs = np.concatenate((state_arr, action_arr), axis=-1)
    labels = np.concatenate((reward_arr, next_state_arr - state_arr), axis=-1)

    return inputs, labels


def model_train():

    inputs, labels = read_data()
    test_size = 0.1
    lr = 0.0001
    num_iterations = 10000

    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs,labels,test_size=0.1, random_state=12315)
    train = BCdataset(inputs_train,labels_train)
    test = BCdataset(inputs_test,labels_test)

    NN_train(train,test,test_size,lr,num_iterations)



if __name__ == "__main__":
    model_train()








