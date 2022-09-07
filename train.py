import os
import numpy as np
import pandas as pd
import argparse
import pickle
import datetime
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from CNN_Model import ConvNet, init_weights

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--data_file",
                        default="generated_data.pickle", type=str,
                        help="File containing generated sequences, labels, and dictionary.")
    parser.add_argument("-ts", "--test_sz",
                        default=0.15, type=float, 
                        help="Decimal representing the percent of the data that will be set aside as the test set.")
    parser.add_argument("-s", "--sampler",
                        default=False, type=bool,
                        help="Whether or not a sampler should be used for dataloader. True recommended for imbalanced dataset.")
    parser.add_argument("-e", "--epochs",
                        default=20, type=int,
                        help="Max number of epochs to train the model.")
    parser.add_argument("-stop", "--early_stop",
                        default=5, type=int,
                        help="Number of epoch to use for early stopping.")
    parser.add_argument("-mn", "--model_name",
                        default='best_model', type=str,
                        help="Filename for saved model.")
    parser.add_argument("-lr", "--lr",
                        default=0.001, type=float,
                        help="Learning rate for the model.")
    parser.add_argument("-d", "--dropout",
                        default=0.1, type=float,
                        help="dropout for the model.")
    return parser.parse_args()

class Seq_data(torch.utils.data.Dataset):

    def __init__(self, dat):
        self.x_data= torch.Tensor(dat[0])
        self.y_data = torch.Tensor(dat[1])
        self.len=len(self.x_data)
      

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def model_acc(test_dat, model, data_name):
    acc = torchmetrics.Accuracy()
    model.eval()
    with torch.no_grad():
        accs = []
        for seqs, labels in test_dat:
            outputs = model(seqs)
            _, predicted = torch.max(outputs.data, 1)
            accs.append(acc(predicted, labels.type(torch.int64)))
        mod_acc = np.mean(accs)
        print('{} Accuracy: {} %'.format(data_name, np.round(mod_acc * 100, 2)))
    return mod_acc

def feat_to_seq(feature_tensor):
    df = pd.DataFrame(np.array(feature_tensor).T)
    new_df = df.idxmax(axis=1)
    new_df.loc[new_df == 0] = 'A'
    new_df.loc[new_df == 1] = 'C'
    new_df.loc[new_df == 2] = 'G'
    new_df.loc[new_df == 3] = 'T'
    seq = "".join(list(new_df))
    return seq

def create_weighted_sampler(train):
    y = train[:][1]
    class_counts = [len(y)-sum(y), sum(y)]
    num_samples = sum(class_counts)
    # labels = [0, 0,..., 0, 1] #corresponding labels of samples

    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[int(y[i])] for i in range(int(num_samples))]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler
if __name__ == '__main__':
    proj_dir = os.path.join(os.getcwd(), '..')
    
    # input arguments
    args = parse_args()
    data_file = args.data_file
    test_sz = args.test_sz
    sampler = args.sampler
    num_epochs = args.epochs
    stop = args.early_stop
    mod_name = args.model_name
    learning_rate = args.lr
    dropout = args.dropout
#     motif_len = args.window_sz
    
    model_name = os.path.join(proj_dir, 'Model', f'{mod_name}.pt')
    data_file_path = os.path.join(proj_dir, 'Data', data_file)
    
    # Read in data
    with open(data_file_path, 'rb') as handle:
        data_dict = pickle.load(handle)
    
    all_seqs = data_dict['seqs']
    all_labels = data_dict['labels']
    seq_dict = data_dict['seq_dict']
    
    t_sz = int(np.ceil(len(all_seqs)*test_sz))
    
    # Load data into dataloader class
    dev = Seq_data((all_seqs[0:t_sz], all_labels[0:t_sz]))
    test =  Seq_data((all_seqs[t_sz:t_sz*2], all_labels[t_sz:t_sz*2]))
    train = Seq_data((all_seqs[t_sz*2:], all_labels[t_sz*2:]))
    
    if sampler:
        sampler = create_weighted_sampler(train)
        train_dat = torch.utils.data.DataLoader(train, batch_size = 100, sampler=sampler)
    else:
        train_dat = torch.utils.data.DataLoader(train, batch_size = 100, shuffle = True)
    test_dat = torch.utils.data.DataLoader(test, batch_size = 100, shuffle = True)
    dev_dat = torch.utils.data.DataLoader(dev, batch_size = 100, shuffle = True)

    train_features, train_labels = next(iter(train_dat))
    print("Percent Positive Class:", sum(train_labels)/len(train_labels))
    
    # create model
    seq_len = len(all_seqs[0][0])
    print(seq_len)
    model = ConvNet(seq_len, 20, dropout, 32, 12, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train model
    total_step = len(train_dat)
    loss_list = []
    acc_list = []
    val_list = []
    es_counter = 0
#     early_stop = 5
    for epoch in range(num_epochs):
        if es_counter >= stop:
            break
        for i, (seqs, labels) in enumerate(train_dat):
            # Run the forward pass
            outputs = model(seqs)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % total_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
                val_list.append(model_acc(dev_dat, model, "Validation"))
                if epoch == 0:
                    best_acc = val_list[-1]
                else:
                    if best_acc < val_list[-1]:
                        torch.save(model.state_dict(), model_name)
                        best_acc = val_list[-1]
                        es_counter = 0
                    else:
                        es_counter += 1
#                 print(best_acc, val_list[-1], es_counter)
                if es_counter >= stop:
                    print("Accuracy has stopped improving, training terminated")
    # test model
    acc = torchmetrics.Accuracy()
    rc = torchmetrics.Recall(average='macro',num_classes = 2)
    pc = torchmetrics.Precision(average='macro',num_classes = 2)
    f1 = torchmetrics.F1Score(average='macro',num_classes = 2)
#     pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=1)
#     roc = torchmetrics.ROC(pos_label=1)
    model.eval()
    seq_preds = []
    with torch.no_grad():
        true = torch.empty(0, dtype=torch.int64)
        pred = torch.empty(0, dtype=torch.int64)
        pos_pred = []
        for seqs, labels in test_dat:
            true = torch.cat((true, labels), 0)
            outputs = model(seqs)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            pred = torch.cat((pred, predicted), 0)
            pos_pred.extend(np.array(outputs.data.T[1]))
            test_sequences = []
            for s in seqs:
                test_sequences.append(feat_to_seq(s))
            seq_preds.extend(zip(test_sequences, labels, pred))
#         print(outputs.data.T[1])
        pos_pred = F.softmax(torch.FloatTensor(pos_pred))
#         print(pos_pred)
        ## Accuracy
        true = true.type(torch.int64)
        accuracy = acc(pred, true)
        precision = pc(pred, true)
        recall = rc(pred, true)
        F1 = f1(pred, true)
        print('{} Accuracy: {} %'.format('Test', np.round(accuracy*100, 2)))
        print('{} F1: {} %'.format('Test', np.round(F1*100, 2)))
        print('{} Recall: {} %'.format('Test', np.round(recall*100, 2)))
        print('{} Precision: {} %'.format('Test', np.round(precision*100, 2)))

        RocCurveDisplay.from_predictions(true, pos_pred)
        
        write_file = os.path.join(proj_dir,'Data', f'preds_{datetime.datetime.now().strftime("%Y%m%d")}.pickle')
        print('Writing data to:', write_file)
    
        with open(write_file, 'wb') as file:
            pickle.dump(seq_preds, file)
            
    
        