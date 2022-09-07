'''
Title: Interpretability Functions
Date: 08/21/22
Author: Jennie Martin
Description: 
'''

import os
import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
from simdna import synthetic as sn
from scipy.stats import wilcoxon
from itertools import combinations
import argparse
from CNN_Model import ConvNet, init_weights

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod", "--model",
                        default="Model/best_model.pt", type=str,
                        help="File path for model")
    parser.add_argument("-p", "--preds",
                        default= "Data/preds.pickle", type=str, 
                        help="File path for model predictions")
    parser.add_argument("-sd", "--seq_dict",
                        default="Data/generated_data.pickle", type=str,
                        help="File path for sequence dictionary")
    parser.add_argument("-m", "--motifs",
                        default=["STAT_known2", "CTCF_known1"], type=str, nargs="*",
                        help="Motif(s) to be included in the tested.")
    parser.add_argument("-bd", "--background_dist",
                        default=20, type=float, nargs="*",
                        help="List of values that represent the distribution of ACGT for the background sequence. Values must add to 1.")
    parser.add_argument("-np", "--num_perturbs",
                        default=5, type=int,
                        help="Number of perturbations for the interpretability functions.")
    return parser.parse_args()

class seq_data(torch.utils.data.Dataset):

    def __init__(self, dat):
        self.x_data= torch.Tensor(dat[0])
        self.y_data = torch.Tensor(dat[1])
        self.len=len(self.x_data)
      

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def test_perturb(test_dat, model):
    model.eval()
    with torch.no_grad():
        for seqs, labels in test_dat:
            outputs = model(seqs)
            labels = labels.type(torch.LongTensor)
    return outputs, labels

def perturb_seq(motifs, seq, seq_dict):
    embeds = seq_dict[seq].embeddings
    new_s = seq
    for m in motifs:
        for e in embeds:
            if m == e.what.getDescription():
                bkgd_sub = sn.ZeroOrderBackgroundGenerator(len(e.what.string), 
                                                discreteDistribution = background_dist).generateBackground()
                new_s = new_s[:e.startPos]+bkgd_sub+new_s[e.startPos+len(e.what.string):]
    new_seq = np.array(pd.get_dummies(pd.Series(list(new_s)))).T
    return [new_seq]

def perturb_seq_order(motifs, seq, seq_dict):
    positions = []
    embeds = seq_dict[seq].embeddings
    # find start and end of each embedding
    for motif in motifs:  
        for e in embeds:
            if motif == e.what.getDescription():
                positions.extend([e.startPos, e.startPos+len(e.what.string)])
    positions = sorted(positions)
    # swap positions of first and second motif
    seq_pieces = [seq[:positions[0]], seq[positions[2]:positions[3]], seq[positions[1]:positions[2]],
                 seq[positions[0]:positions[1]], seq[positions[3]:]]

    new_seq = ''.join(seq_pieces)
    new_seq = np.array(pd.get_dummies(pd.Series(list(new_seq)))).T

    return [new_seq]

def perturb_seq_dist(motifs, seq, seq_dict):
    positions = []
    emb_sizes = []
    embeds = seq_dict[seq].embeddings
    for motif in motifs:  
        for e in embeds:
            if motif == e.what.getDescription():
                emb_sizes.append(len(e.what.string))
                positions.extend([e.startPos, e.startPos+len(e.what.string)])
    positions = sorted(positions)
    # generate new positions
    new_positions = []
    motif1_len = positions[1] - positions[0]
    motif2_len = positions[3] - positions[2]
    new_positions.append(random.randint(0, len(seq) - motif1_len - motif2_len))
    new_positions.append(random.randint(0, len(seq) - motif2_len))
    new_positions = sorted(new_positions)
    
    escape = 0 # make sure second motif doesn't overwrite the first
    while new_positions[1] - new_positions[0] < motif1_len:
        new_positions[1] = random.randint(new_positions[0] + motif1_len, len(seq) - motif2_len)
        escape += 1
        if escape > 2:
            print(new_positions, motif1_len, len(seq) - max(emb_sizes)) # for trouble-shooting
    # remove motifs from base sequence and insert them back in new positions
    base_seq = ''.join([seq[:positions[0]], seq[positions[1]:positions[2]], seq[positions[3]:]])
    new_seq = ''.join([base_seq[:new_positions[0]], seq[positions[0]:positions[1]], base_seq[new_positions[0]:]])
    new_seq = ''.join([new_seq[:new_positions[1]], seq[positions[2]:positions[3]], new_seq[new_positions[1]:]])
    
    assert(len(new_seq) == len(seq))
    new_seq = np.array(pd.get_dummies(pd.Series(list(new_seq)))).T

    return [new_seq]

def perturbed_pipeline(motifs, seq, true_label, seq_dict, model):
    ps = perturb_seq(motifs, seq, seq_dict)
    perturb_dat = torch.utils.data.DataLoader(seq_data((ps, [true_label])), batch_size = 1)
    return test_perturb(perturb_dat, model)

def test_perturbed_loss(model, test_output, test_type, motifs, seq_dict, num_perturbs = 5):
    """ 
    Input:
        model: trained model
        test_output: output of test_model function containing [(sequence, true label, predicted label)]
        test_type: "presense", "order", "distance"
        motifs: char string of motif (ex. 'CTCF_known1')
        seq_dict: dictionary mapping char string of sequence to it's corresponing simdna GeneratedSequence object. 
        num_perturbs: number of times to perturb the sequence
    Returns: difference between loss of original function and the average loss of function when sequences are perturbed
    """
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    with torch.no_grad():
        loss_diff = []
        if test_type == 'presence': 
            perturb_func = perturb_seq
            motifs = [motifs[0]] #only test one motif for presence
        elif test_type == 'order':
            perturb_func = perturb_seq_order
        elif test_type == 'distance':
            perturb_func = perturb_seq_dist
        else:
            raise("Test type not supported. Supported types include presence, order, and distance")

        for seq_info in test_output:
            p_loss = []
            seq, tl, pl = seq_info
            embeds = set([e.what.getDescription() for e in seq_dict[seq].embeddings])
            if set(motifs).issubset(embeds):
            # get original loss
                orig_seq = [np.array(pd.get_dummies(pd.Series(list(seq)))).T]
                orig_dat = torch.utils.data.DataLoader(seq_data((orig_seq, [tl])))
                outputs, labels = test_perturb(orig_dat, model)
                orig_loss = criterion(outputs, labels)
            # get average perturbed loss
                for p in range(num_perturbs):
                    ps = perturb_func(motifs, seq, seq_dict)
                    perturb_dat = torch.utils.data.DataLoader(seq_data((ps, [tl])))
                    perturb_loss = criterion(*test_perturb(perturb_dat, model))
                    p_loss.append(perturb_loss)
                new_loss = np.mean(p_loss)
                loss_diff.append(float(orig_loss-new_loss))
    return loss_diff #, zip((orig_losses, new_losses))


def test_perturbed_interact_loss(model, test_dat, motifs, seq_dict, num_perturbs = 5):
    """ 
    Input:
        model: trained model
        test_dat: output of test_model function containing [(sequence, true label, predicted label)]
        motifs: list of two char string of motifs (ex. ['CTCF_known1', 'STAT_known2'])
        seq_dict: dictionary mapping char string of sequence to it's corresponing simdna GeneratedSequence object. 
        num_perturbs: number of times to perturb the sequence
    Returns: difference between loss of original function and the average loss of function when sequences are perturbed
    """
    loss_diff = []
    motif_A = motifs[0]
    motif_B = motifs[1]
    
    criterion = nn.CrossEntropyLoss()
    
    for seq_info in test_dat:
        p_loss = []
        seq, true_label, pred_label = seq_info
        embeds = [e.what.getDescription() for e in seq_dict[seq].embeddings]
        # only perturb seq if all test motifs occur in the given seq
        if motif_A in embeds and motif_B in embeds:
            # get original loss
            orig_seq = [np.array(pd.get_dummies(pd.Series(list(seq)))).T]
            orig_dat = torch.utils.data.DataLoader(seq_data((orig_seq, [true_label])))
            orig_out, label = test_perturb(orig_dat, model)

            A_out = []
            B_out = []
            AB_out = []
            # get perturbed loss
            for p in range(num_perturbs):
                A_out.append(perturbed_pipeline([motif_A], seq, true_label, seq_dict, model)[0])
                B_out.append(perturbed_pipeline([motif_B], seq, true_label, seq_dict, model)[0])
                AB_out.append(perturbed_pipeline([motif_A, motif_B], seq, true_label, seq_dict, model)[0])
            outputs = (orig_out + (torch.mean(torch.stack(B_out), dim=0) - orig_out) + 
                       (torch.mean(torch.stack(A_out), dim=0) - orig_out))
            model.eval()
            with torch.no_grad():
                loss = criterion(outputs, label)
                loss_AB = criterion(torch.mean(torch.stack(AB_out), dim=0), label)
            diff = loss_AB -loss
            loss_diff.append(float(diff))
    return loss_diff

def test_significance(model, preds, seq_dict, motifs, num_perturbs):
    results = [['Test', 'effect_size', 'count', 'avg_loss_diff', 'p_val']]
#     sig_results = []
    for motif in motifs:
        losses = test_perturbed_loss(model, preds, 'presence', [motif], seq_dict, num_perturbs)
        avg_loss = np.mean(losses)
        effect = sum(losses)
        p_val = wilcoxon(losses)[1]
        count = len(losses)
        results.append([f"presence_of_{motif}", effect, count, avg_loss, p_val])
        print(f"presence_of_{motif}", effect, count, avg_loss, p_val)

    combo_feats = list(combinations(motifs, 2))
    
    for mtfs in combo_feats:
        losses_o = test_perturbed_loss(model, preds, 'order', list(mtfs), seq_dict, num_perturbs)
        avg_loss = np.mean(losses_o)
        effect = sum(losses_o)
        count = len(losses_o)
        p_val = wilcoxon(losses_o)[1]
        results.append([f"order_of_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val])
        print(f"order_of_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val)
        
        losses_d = test_perturbed_loss(model, preds, 'distance', list(mtfs), seq_dict, num_perturbs)
        avg_loss = np.mean(losses_d)
        effect = sum(losses_d)
        count = len(losses_d)
        p_val = wilcoxon(losses_d)[1]
        results.append([f"dist_btwn_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val])
        print(f"dist_btwn_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val)
        
        losses_i = test_perturbed_interact_loss(model, preds, list(mtfs), seq_dict, num_perturbs)
        avg_loss = np.mean(losses_i)
        effect = sum(losses_i)
        count = len(losses_i)
        p_val = wilcoxon(losses_i)[1]
        results.append([f"interact_btwn_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val])
        print(f"interact_btwn_{mtfs[0]}_&_{mtfs[1]}", effect, count, avg_loss, p_val)
        
    return results

if __name__ == '__main__':
    proj_dir = os.path.join(os.getcwd(), '..')
    # input arguments
    args = parse_args()
    model_path = args.model
    pred_path = args.preds
    seq_dict_path = args.seq_dict
    motifs = args.motifs
    background_vals = args.background_dist
    num_perturbs = args.num_perturbs
    
    model = ConvNet(200, 20, 0.1, 32, 12, 2)
    model.load_state_dict(torch.load(os.path.join(proj_dir, model_path)))
    with open(os.path.join(proj_dir, pred_path), 'rb') as handle:
        preds = pickle.load(handle)
    with open(os.path.join(proj_dir, seq_dict_path), 'rb') as handle:
        data_dict = pickle.load(handle)
    seq_dict = data_dict['seq_dict']
    
    background_dist = dict(zip(['A','C','G','T'], background_vals))
    
    results = test_significance(model, preds, seq_dict, motifs, num_perturbs)
    
    results_df = pd.DataFrame(results, columns = ['test', 'effect_size', 'count', 'avg_loss', 'p_value'])
    results_df.to_csv(os.path.join(proj_dir,'Output', 'interpret_results.csv'), index=False)