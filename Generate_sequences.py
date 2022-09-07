'''
Title: Generate Sequences
Date: 08/21/22
Author: Jennie Martin
Description: 
'''

import simdna
from simdna import synthetic as sn
import random
import numpy as np
import pandas as pd
import argparse
import os
import pickle
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--motifs",
                        default=["STAT_known2", "CTCF_known1"], type=str, nargs="*",
                        help="Motif(s) to be included in the generated sequence.")
    parser.add_argument("-pm", "--pos_motifs",
                        default=["STAT_known2"], type=str, nargs="*",
                        help="List of motif(s) that define the postive class. If order is a factor, list the motifs in the order they should appear in the sequence")
    parser.add_argument("-t", "--gen_type",
                        default='presence', type=str, nargs="*",
                        help="Defines what is important to the positive class- presence, order, or distance of the motifs. If order is selected, presence will also be evaluated. If distance is selected, presence and order will be evaluated.")
    parser.add_argument("-mnd", "--min_dist",
                        default=0, type=str,
                        help="The minimum distance the motifs can be from each other to be a part of the positive class")
    parser.add_argument("-mxd", "--max_dist",
                        default=20, type=str,
                        help="The maximum distance the motifs can be from each other to be a part of the positive class")
    parser.add_argument("-bd", "--background_dist",
                        default=20, type=float, nargs="*",
                        help="List of values that represent the distribution of ACGT for the background sequence. Values must add to 1.")
    parser.add_argument("-n", "--num_seqs",
                        default=100, type=int,
                        help="Number of sequences to generate")
    parser.add_argument("-l", "--seq_length",
                        default=200, type=int,
                        help="Length of sequences")
    return parser.parse_args()

def generate_sequences(num_seqs, seq_len, background_dist, motif_names,  min_motifs, max_motifs):
    # load in motifs and create motifs
    loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                           pseudocountProb=0.001)
    position_generator = sn.UniformPositionGenerator()
    embedders = [sn.SubstringEmbedder(sn.PwmSamplerFromLoadedMotifs(
                 loaded_motifs, motif_name),
                 position_generator, name=motif_name)
                 for motif_name in motif_names]
    # initalize generator to pick random set of motifs in each sequence
    min_selected_motifs = min_motifs
    max_selected_motifs = max_motifs
    quantity_generator = sn.UniformIntegerGenerator(min_selected_motifs,
                                                 max_selected_motifs)
    combined_embedder = [sn.RandomSubsetOfEmbedders(
                         quantity_generator, embedders)]
    embed_in_background = sn.EmbedInABackground(
        sn.ZeroOrderBackgroundGenerator(
         seq_len, discreteDistribution = background_dist),
        combined_embedder)
    generated_sequences = tuple(sn.GenerateSequenceNTimes(
        embed_in_background, num_seqs).generateSequences())
    embedding_arr = []
    for generated_seq in generated_sequences:
        embedding_dict = {}
        for x in generated_seq.embeddings:
            embedding_dict[x.what.getDescription()] = x.startPos
        embedding_arr.append(embedding_dict)
    
    return generated_sequences, embedding_arr

def gen_seqs_w_position(num_seqs, seq_len, background_dist, motif_names, pair, min_dist, max_dist, min_motifs, max_motifs):
    # load in motifs and create motifs
    loaded_motifs = sn.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH,
                           pseudocountProb=0.001)
    position_generator = sn.UniformPositionGenerator()
    
    separationGenerator=sn.MinMaxWrapper(
        sn.PoissonQuantityGenerator(20),
        theMin=min_dist,
        theMax=max_dist) 
#     separationGenerator= sn.UniformIntegerGenerator(10, 30)
    embedders1 = [sn.SubstringEmbedder(sn.PwmSamplerFromLoadedMotifs(
                 loaded_motifs, motif_name),
                 position_generator, name=motif_name)
                 for motif_name in motif_names]
   
    # initalize generator to pick random set of motifs in each sequence
    quantity_generator1 = sn.UniformIntegerGenerator(min_motifs, max_motifs)
    embedder1 = [sn.RandomSubsetOfEmbedders(quantity_generator1, embedders1)]
    embed_in_background1 = sn.EmbedInABackground(sn.ZeroOrderBackgroundGenerator(
        seq_len, discreteDistribution = background_dist), embedder1)
    generated_sequences1 = list(sn.GenerateSequenceNTimes(embed_in_background1, 
                                                           int(np.ceil(num_seqs*.55))).generateSequences())
    # create additional sequences that insure the distance constraint is satisfied
    kwargs={'loadedMotifs':loaded_motifs}
    motif1Generator=sn.PwmSamplerFromLoadedMotifs(motifName=pair[0],**kwargs)
    motif2Generator=sn.PwmSamplerFromLoadedMotifs(motifName=pair[1],**kwargs)
    motif1Embedder=sn.SubstringEmbedder(substringGenerator=motif1Generator)
    motif2Embedder=sn.SubstringEmbedder(substringGenerator=motif2Generator)
    
    embedders2 = [sn.EmbeddableEmbedder(embeddableGenerator=sn.PairEmbeddableGenerator(
                            embeddableGenerator1=motif1Generator, embeddableGenerator2=motif2Generator,
                            separationGenerator=separationGenerator, name = 'pair'))]
    quantity_generator2 = sn.FixedQuantityGenerator(1)
    embedder2 = [sn.RandomSubsetOfEmbedders(quantity_generator2, embedders2)]
    embed_in_background2 = sn.EmbedInABackground(sn.ZeroOrderBackgroundGenerator(
         seq_len, discreteDistribution = background_dist), embedder2)
    generated_sequences2 = list(sn.GenerateSequenceNTimes(embed_in_background2, 
                                                           int(np.floor(num_seqs*.45))).generateSequences())
    # combine and shuffle sequences
    generated_sequences1.extend(generated_sequences2)
    random.shuffle(generated_sequences1)
    generated_sequences = tuple(generated_sequences1)
    
    #extract list of embeddings for each sequence
    embedding_arr = []
    for generated_seq in generated_sequences:
        embedding_dict = {}
        for x in generated_seq.embeddings:
            embedding_dict[x.what.getDescription()] = x.startPos
        embedding_arr.append(embedding_dict)
    
    return generated_sequences, embedding_arr

if __name__ == '__main__':
    proj_dir = os.getcwd()
    # input arguments
    args = parse_args()
    class_type = args.gen_type
    motif_names = args.motifs
    seq_len = args.seq_length
    num_seqs = args.num_seqs
    min_dist = args.min_dist
    max_dist = args.max_dist
    background_vals = args.background_dist
    pos_motifs = args.pos_motifs
    
#     assert np.round(sum(background_vals),2) == 1.0, f'Background distribution percentages sum to {sum(background_vals)}. They must sum to 1.'
    background_dist = dict(zip(['A','C','G','T'], background_vals))
    
    # generate sequences
    if class_type == 'presence': # gen_seqs_w_position ensures the dist conditions occur in a good amount of the sequences
        gs, ea = gen_seqs_w_position(num_seqs, seq_len, background_dist, motif_names,  min_dist, max_dist, 0, len(motif_names))
    else:
        gs, ea = generate_sequences(num_seqs, seq_len, background_dist, motif_names,  0, len(motif_names))
    # create sequence mappings and generate labels:
    seqs = []
    seq_dict = {}
    
    seq_label = np.zeros([len(gs)])
    for i in range(len(gs)):
        seq_dict[gs[i].seq] = gs[i] # create dict of sequences
        seqs.append(np.array(pd.get_dummies(pd.Series(list(gs[i].seq)))).T) # create features
        #create labels
        cnt = 0
        for m in pos_motifs:
            if m in ea[i].keys():
                cnt += 1
        if cnt == len(pos_motifs):
            if class_type == 'distance':
                x = (ea[s][pos_motif[1]] - ea[s][pos_motif[0]] + len(gs[i].seq))
                if x >= min_dist and x <= max_dist:
                    seq_label[i] = 1
            elif class_type == 'order':
                if ea[i][pos_motif[0]] < ea[i][pos_motif[1]]:
                    seq_label[i] = 1
            else:
                seq_label[i] = 1
    data = {'seqs':seqs, 'labels':seq_label, 'seq_dict':seq_dict}
    
    # write out files
    write_file = os.path.join(proj_dir, "..",'Data', f'generated_data_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.pickle')
    print('Writing data to:', write_file)
                              
    with open(write_file, 'wb') as file:
        pickle.dump(data, file)
    