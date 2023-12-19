# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MFP2TCANet
from datasets import training_data
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor
import joblib


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    device = torch.device('cuda:0')
    setup_seed(1234)
    x_feature1, x_feature2, y = training_data()
    x = torch.cat((x_feature1, x_feature2), dim=1)  # using multi feature + pca

    mean_list1, mean_list2, mean_list3 = [], [], []
    mean_list5, mean_list6, mean_list7 = [], [], []

    split_shape = y.shape
    train_index = np.array(range(0, int(np.ceil(split_shape[0]*0.75))))
    test_index = np.array(range(int(np.ceil(split_shape[0]*0.75)), split_shape[0]))
    train_x = x[train_index].to(device)
    train_y = y[train_index].to(device)
    test_x = x[test_index].to(device)
    test_y = y[test_index].to(device)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    model = MFP2TCANet()
    model.to(device)
    steps = 0
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = torch.nn.MSELoss()
    clip = -1
    for epoch in range(10):
        train_loss = 0
        model.train()
        i = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            feature1 = data[:, 0, :]
            feature2 = data[:, 1, :]
            feature1 = feature1.view(-1, 1, 100)
            feature2 = feature2.view(-1, 1, 100)
            cat, output = model(feature1, feature2)
            loss = loss_fc(output, target)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss
            i += 1
        print('\nepoch {}: train set: Average loss: {:.4f},\n'.format(epoch,  train_loss / i))
        if epoch % 4 == 0:
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    model.eval()
    cat_train_list = []
    cat_lstm_train_list = []

    for batch_idx, (data, target) in enumerate(train_loader):
        feature1 = data[:, 0, :]
        feature2 = data[:, 1, :]
        feature1 = feature1.view(-1, 1, 100)
        feature2 = feature2.view(-1, 1, 100)
        cat, output = model(feature1, feature2)
        cat_train_list.extend(cat.detach().cpu().numpy())

    ########################################
    cat_test_list = []
    cat_lstm_test_list = []
    output_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        feature1 = data[:, 0, :]
        feature2 = data[:, 1, :]
        feature1 = feature1.view(-1, 1, 100)
        feature2 = feature2.view(-1, 1, 100)
        cat, output = model(feature1, feature2)
        cat_test_list.extend(cat.detach().cpu().numpy())
        output_list.extend(output.detach().cpu().numpy())
    output_list = np.array(output_list)
    train_y = np.array(train_y.detach().cpu().numpy()).ravel()
    test_y = np.array(test_y.detach().cpu().numpy()).ravel()

    cat_train_list = np.array(cat_train_list)
    cat_test_list = np.array(cat_test_list)

    KNR = KNeighborsRegressor(n_neighbors=5)

    KNR.fit(cat_train_list, train_y)
    knr_y_pred = KNR.predict(cat_test_list)
    joblib.dump(KNR, 'knr_y.joblib')
    # one y, 25ms, for better visualization
    rec_test_y = np.zeros(len(test_y)//10)
    rec_output_list = np.zeros(len(test_y)//10)
    rec_knr_y_pred = np.zeros(len(test_y)//10)

    for i in range(len(test_y)//10):
        rec_test_y[i] = np.mean(test_y[10*i:10*i+10])
        rec_knr_y_pred[i] = np.mean(knr_y_pred[10*i:10*i+10])
    plt.figure(figsize=(5,3))
    plt.plot(rec_knr_y_pred)
    plt.plot(rec_test_y)
    plt.savefig("test.png")






