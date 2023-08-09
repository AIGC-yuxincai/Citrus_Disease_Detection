import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from HsiDataset import HsiDataset,HsiDatasetTest
from RgbDataset import RgbDataset
import os


isExists = lambda path: os.path.exists(path)

LR = 1e-2
BATCHSZ = 8
NUM_WORKERS = 12
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def test(model, criterion, dataLoader,log = False):
    model.eval()
    evalLoss, correct = [], 0

    if log == True:
        with open('True_False_classification_result.txt', 'a') as f:
            f.write('True label：  ' + 'Predict label：  ' + 'index:    \n')
        if f.closed:
            pass
        else:
            print('File is still open.')

    for step, data in enumerate(dataLoader):
        spectra = data[0]
        target = data[1]
        spectra, target = spectra.to(DEVICE), target.to(DEVICE)
        out = model(spectra)
        # target = target.view(1, -1)
        loss = criterion(out, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(out, dim=-1)

        if log == True:
            complet_name = data[2]
            all_labels = torch.cat((target.unsqueeze(1), pred.unsqueeze(1)), dim=1)
            with open("True_False_classification_result.txt", "a") as f:
                for i, row in enumerate(all_labels):
                    line = "     ".join(str(x.item()) for x in row)
                    f.write(f"{line}     {complet_name[i]}\n")
            if f.closed:
                pass
            else:
                print('File is still open.')

        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)

def main():
    data = "data_CARS"
    root_dir =  "./dataset/" + data + "/"
    imput_size = 224

    if data == "data_RGB":
        test_data = RgbDataset(root_dir + "test", imput_size)
        testLoader = DataLoader(test_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    elif data == "data_CARS" or data == "data_Original":
        test_data = HsiDatasetTest(root_dir + "test", imput_size)
        testLoader = DataLoader(test_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    else:
        test_data = HsiDataset(root_dir + "test", imput_size)
        testLoader = DataLoader(test_data, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)

    # load save model
    checkpoint = torch.load('./checkpoint/Our_model_CARS.pth',map_location = 'cuda:0')
    model = checkpoint['model']

    criterion = nn.CrossEntropyLoss()
    # if log = True,output classification result
    if data == "data_CARS":
        test_acc, test_evalLoss = test(model, criterion=criterion, dataLoader=testLoader,log=True)
    else:
        test_acc, test_evalLoss = test(model, criterion=criterion, dataLoader=testLoader, log=False)
    print("The accuracy of the model on the verification set is{},evalLoss为{}".format(test_acc,test_evalLoss))

if __name__ == '__main__':
    main()
    print('*'*5 + 'FINISH' + '*'*5)
