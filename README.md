# Model-Agnostic-Interpretation-of-Learned-Models-for-Genomic-Sequences

## Generating Sequences
To test the interpretability functions, we needed a dataset with a known function we could try to learn in the modeling phase and then describe in the interpretability phase. To ensure the correct function built into the data, we generated and labeled our own data.     

To generate sequences you can call the Generate_sequences.py file and use the following arguments:
- num_seqs: number of sequences to generated.
- seq_length: number of bases in the sequences generated.
- gen_type: The gen_type argument refers to what kind of sequence is being generated. The type can either be presence, order, or distance and the value will determine what information is built into the sequences and how they are labeled.
- background_dist: a list of 4 percentages, summing to 1, that represent the background distribution of A, C, G, T in the sequence before motifs are inserted. There are additional ways to create a background distribution built into the simdna package, but it is not currently used in this code.
- motifs: list specifying the names of the motifs you want inserted into the sequences. A list of motifs can be found here [https://github.com/kundajelab/simdna/tree/8211dbab2030a3d00275a254f25c13a67c640e2d/simdna/resources]
- pos_motifs: list specifying the motifs that will dictate a positive class for a sequence. If you want the presence of STAT_known2 in a sequence to indicate a positive outcome STAT_known2 is the pos_motif. Any number of motifs can be provided for sequences that only encode the presence of a motif, but for sequences that will encode order or distance between motifs, two motifs must be provided.
- min_dist: If distance is important, what is the minimum distance the two motifs can appear from each other?
- max_dist: If distance is important, what is the maximum distance the two motifs can appear from each other?    

Example of how to run this step:

python Generate_sequences.py 

For more information about the simdna package, please see this GitHub page [https://github.com/kundajelab/simdna/tree/8211dbab2030a3d00275a254f25c13a67c640e2d]


## Training the Models


## Interpreting the Models
