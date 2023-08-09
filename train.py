import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from HsiDataset import HsiDataset
from RgbDataset import RgbDataset
import os
import argparse
from visdom import Visdom

from Model.Resnet12 import Resnet12
from Model.Resnet18 import Resnet18
from Model.Resnet34 import resnet34
from Model.Resnet50 import Resnet50
from Model.EfficientnetV2 import efficientnetv2_s
from Model.VIT import my_vit_patch16_224
from Model.our_model import Our_model

# You need to start visdom first：python -m visdom.server

isExists = lambda path: os.path.exists(path)
LR = 1e-2
BATCHSZ = 8
NUM_WORKERS = 12
data = "data_CARS"
model_name = "Our_model"
SEED = 971104   #random seed
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()

def train(model, criterion, optimizer, dataLoader):

    model.train()
    model.to(DEVICE)
    trainLoss = []
    for step, data in enumerate(dataLoader):
        spectra = data[0]
        target = data[1]
        spectra, target = spectra.to(DEVICE), target.to(DEVICE)
        out = model(spectra)
        loss = criterion(out, target)

        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, float(np.mean(trainLoss))

def test(model, criterion, dataLoader,log = False):
    model.eval()
    evalLoss, correct = [], 0


    for step, data in enumerate(dataLoader):
        spectra = data[0]
        target = data[1]
        spectra, target = spectra.to(DEVICE), target.to(DEVICE)
        out = model(spectra)
        loss = criterion(out, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(out, dim=-1)

        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)

def main():
    # load data
    root_dir =  "./dataset/" + data + "/"
    imput_size = 224
    nc = 4

    bands = 44  # spectral image of channels
    if data == "data_CARS":
        bands = 44
    elif data == "data_CARS_35":
        bands = 35
    elif data == "data_Original":
        bands = 224
    elif data == "data_RGB":
        bands = 3

    if data == "data_RGB":
        train_data = RgbDataset(root_dir + "train", imput_size, transform=True)   #start dataAugument
        valid_data = RgbDataset(root_dir + "val", imput_size)
        test_data = RgbDataset(root_dir + "test", imput_size)
    else:
        train_data = HsiDataset(root_dir + "train", imput_size, transform=True)   #start dataAugument
        valid_data = HsiDataset(root_dir + "val", imput_size)
        test_data = HsiDataset(root_dir + "test", imput_size)

    trainLoader = DataLoader(train_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    validLoader = DataLoader(valid_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(test_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)

    if model_name == "Resnet12":
        model = Resnet12(num_classes=nc, channels=bands)
    elif model_name == "Resnet18":
        model = Resnet18(num_classes=nc, channels=bands)
    elif model_name == "Resnet34":
        model = resnet34(num_classes=nc, channels=bands)
    elif model_name == "Resnet50":
        model = Resnet50(num_classes=nc,channels=bands)
    elif model_name == "EfficientnetV2":
        model = efficientnetv2_s(num_classes=nc,channels=bands)
    elif model_name == "Vit":
        model = my_vit_patch16_224(num_classes=nc,channels=bands)
    else:
        model_name = "Our_model"
        model = Our_model(num_classes=nc, channels=bands)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)

    train_losses = []
    test_losses = []
    train_accs = []

    # Save the model. Here is the benchmark accuracy rate
    best_acc = 0.3

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=validLoader)

        if acc > best_acc and (epoch > (EPOCHS-30)):
            best_acc = acc
            tail = int(acc*100)
            model_out_path = "checkpoint/" + "{}_model_epoch_{}_acc_{}.pth".format(model_name,epoch+1,tail)
            state = {"epoch": epoch+1, "model": model}
            if not os.path.exists("checkpoint/"):
                os.makedirs("checkpoint/")
            torch.save(state, model_out_path)
            print("Best Checkpoint saved to {}".format(model_out_path))

        train_losses.append(trainLoss)
        test_losses.append(evalLoss)
        train_accs.append(acc)

        viz.line([[trainLoss, evalLoss]], [epoch], win='train&test loss', update='append')
        viz.line([acc], [epoch], win='accuracy', update='append')
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        scheduler.step()

    test_acc, test_evalLoss = test(model, criterion=criterion, dataLoader=testLoader,log=True)
    print("The accuracy of the model on the verification set is{},evalLoss为{}".format(test_acc,test_evalLoss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrain HsiCnn')

    parser.add_argument('--name', type=str, default='fruit disease',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=150,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--data', type=str, default='data_CARS',
                        help='data_CARS or data_Original or data_RGB')
    parser.add_argument('--model_name', type=str, default='Our_model',
                        help='model name')


    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    viz.line([[0., 0.]], [0.], win='train&test loss', opts=dict(title='train&test loss',
                                                                legend=['train_loss', 'test_loss']))
    viz.line([0.,], [0.,], win='accuracy', opts=dict(title='accuracy',
                                                     legend=['accuracy']))
    print('*'*5 + 'PreTrained By {}'.format(datasetName) + '*'*5)
    main()
    print('*'*5 + 'FINISH' + '*'*5)
