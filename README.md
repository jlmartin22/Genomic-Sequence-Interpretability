# Model-Agnostic-Interpretation-of-Learned-Models-for-Genomic-Sequences

## Generating Sequences
To test the interpretability functions, we needed a dataset with a known function we could try to learn in the modeling phase and then describe in the interpretability phase. To ensure the correct function built into the data, we generated and labeled our own data.     

#### Input Arguments
To generate sequences you can call the Generate_sequences.py file and use the following arguments:
- num_seqs: number of sequences to generated.
- seq_length: number of bases in the sequences generated.
- gen_type: The gen_type argument refers to what kind of sequence is being generated. The type can either be presence, order, or distance and the value will determine what information is built into the sequences and how they are labeled.
- background_dist: a list of 4 percentages, summing to 1, that represent the background distribution of A, C, G, T in the sequence before motifs are inserted. There are additional ways to create a background distribution built into the simdna package, but it is not currently used in this code.
- motifs: list specifying the names of the motifs you want inserted into the sequences. A list of motifs can be [found here](https://github.com/kundajelab/simdna/tree/8211dbab2030a3d00275a254f25c13a67c640e2d/simdna/resources).
- pos_motifs: list specifying the motifs that will dictate a positive class for a sequence. If you want the presence of STAT_known2 in a sequence to indicate a positive outcome STAT_known2 is the pos_motif. Any number of motifs can be provided for sequences that only encode the presence of a motif, but for sequences that will encode order or distance between motifs, two motifs must be provided.
- min_dist: If distance is important, what is the minimum distance the two motifs can appear from each other?
- max_dist: If distance is important, what is the maximum distance the two motifs can appear from each other?    

Example of how to run this step:   
'''
python Generate_sequences.py --num_seqs 100000 --seq_length 200 --gen_type order --background_dist 0.15 0.2 0.3 0.35 --motifs STAT_known2 CTCF_known1 --pos_motifs STAT_known2 CTCF_known1
'''
#### Output
This script will produce a generated_data.pickle file in a Data folder whose parent directory is one folder up from the code. This pickle file is a dictionary that contains an array of sequences (seqs), an array of corresponding labels (labels), and a sequence dictionary (seq_dict) that maps the string representation of the sequence to the generated_sequence class object that contains more information about the sequence. For more information about the generated_sequence class please see the simdna package.    

For more information about the simdna package, [please see this GitHub page](https://github.com/kundajelab/simdna/tree/8211dbab2030a3d00275a254f25c13a67c640e2d).   


## Training the Model
The goal of this work is to be model agnostic, but the model that was used for the current tests is a convolutional neural net that uses the architecture described by Eraslan et al. in [Deep learning: new computational modelling techniques for genomics](https://www.nature.com/articles/s41576-019-0122-6). The training script was created with this model in mind, but can likely be tweaked to work for other models.

#### Input arguments
- data_file: the path to the data_file created in the Generate_sequences.py script. The seqs and labels objects in this file will be used, the seq_dict is not used in this step.
- test_sz: decimal representing the percent of the data the will be reserved as the test set. Please note that there will also be a validation set that will use the same percentage, thus the train set will be '''1-(test_sz*2)'''.
- sampler: True if you want the training dataloader class to sample using weighted probabilities. Recommended there is class imbalance in the training data.
- epochs: max number of epoch for the model to run.
- early_stop: how many epochs can elapse without the validation accuracy improving.
- model_name: name to save best model to.
- lr: learning rate for model.
- dropout: dropout rate for model layers.    
Example of how to run this step:   
'''
python train.py --data_file generated_data_20220824-2319.pickle --test_sz .15 --sampler True --epochs 50 --early_stop 5 --model_name best_model --lr .0001 --dropout .1
'''   

#### Output
The training script outputs a trained model and a file saved as Data/preds_{cur_date}.pickle with the test sequences and their corresponding true and prediction labels. Both are needed for the interpretability script.

## Interpreting the Models

#### Input arguments

Example of how to run this step:   
'''
python interpretability_functions.py --model Model/best_model.pt --preds Data/preds_20220901.pickle --seq_dict Data/generated_data_20220824-2319.pickle --motifs STAT_known2 CTCF_known1 --background_dist 0.15 0.2 0.3 0.35 --num_perturbs 5
'''   
 #### Outputs:
This script will output a csv with all of the interpretability information, including the total effect size, the count of sequences tested, the average change in loss per sequence, and the p value for each test.
