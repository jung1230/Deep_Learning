import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
import os, sys

# Add DLStudio-2.5.2 and AdversarialLearning to sys.path so Python can find DLStudio and AdversarialLearning
current_dir = os.getcwd()
print("current_dir = %s" % current_dir)

DLStudio_dir = os.path.join(current_dir, "../DLStudio-2.5.3")
sys.path.append(DLStudio_dir)
# Transformer_dir = os.path.join(current_dir, "../DLStudio-2.5.3")
# sys.path.append(Transformer_dir)

from DLStudio import *
from Transformers import *


# this code is borrowed from DLstudio

import random
import numpy
import torch
import os, sys

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


dataroot = "./../data/DataForXformer/"


data_archive =  "en_es_xformer_8_10000.tar.gz"                     ## for debugging only
# data_archive =  "en_es_xformer_8_90000.tar.gz" # this takes so long to train


max_seq_length = 10

embedding_size = 256        
#embedding_size = 128
#embedding_size = 64        

num_basic_encoders = num_basic_decoders = num_atten_heads = 4     
#num_basic_encoders = num_basic_decoders = num_atten_heads = 2    

#optimizer_params = {'beta1' : 0.9,  'beta2': 0.98,  'epsilon' : 1e-9}
optimizer_params = {'beta1' : 0.9,  'beta2': 0.98,  'epsilon' : 1e-6}

num_warmup_steps = 4000

masking = True                     ## for better results
#masking = False

dls = DLStudio(
                dataroot = dataroot,
                path_saved_model = {"encoder_FG" : "./saved_encoder_FG", 
                                    "decoder_FG" : "./saved_decoder_FG", 
                                    "embeddings_generator_en_FG" : "./saved_embeddings_generator_en_FG",
                                    "embeddings_generator_es_FG" : "./saved_embeddings_generator_es_FG",
                                   },
                batch_size = 50,
                use_gpu = True,
                epochs = 40,
              )

xformer = TransformerFG( 
                        dl_studio = dls,
                        dataroot = dataroot,
                        data_archive = data_archive,
                        max_seq_length = max_seq_length,
                        embedding_size = embedding_size,
                        save_checkpoints = True,
                        num_warmup_steps = num_warmup_steps,
                        optimizer_params = optimizer_params,
          )

master_encoder = TransformerFG.MasterEncoder(
                                  dls,
                                  xformer,
                                  num_basic_encoders = num_basic_encoders,
                                  num_atten_heads = num_atten_heads,
                 )    


master_decoder = TransformerFG.MasterDecoderWithMasking(
                                  dls,
                                  xformer, 
                                  num_basic_decoders = num_basic_decoders,
                                  num_atten_heads = num_atten_heads,
                                  masking = masking
                 )


number_of_learnable_params_in_encoder = sum(p.numel() for p in master_encoder.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Master Encoder: %d" % number_of_learnable_params_in_encoder)

number_of_learnable_params_in_decoder = sum(p.numel() for p in master_decoder.parameters() if p.requires_grad)
print("\nThe number of learnable parameters in the Master Decoder: %d" % number_of_learnable_params_in_decoder)

if masking:
    xformer.run_code_for_training_TransformerFG(dls,master_encoder,master_decoder,display_train_loss=True,
                                                                                     checkpoints_dir="checkpoints_with_masking_FG")
else:
    xformer.run_code_for_training_TransformerFG(dls,master_encoder,master_decoder,display_train_loss=True,
                                                                                        checkpoints_dir="checkpoints_no_masking_FG")

#import pymsgbox
#response = pymsgbox.confirm("Finished training.  Start evaluation?")

#if response == "OK": 
xformer.run_code_for_evaluating_TransformerFG(master_encoder, master_decoder, 'myoutput.txt')


# this code is borrowed from DLstudio

import random
import numpy
import torch
import os, sys

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

dataroot = "./../data/DataForXformer/"


data_archive =  "en_es_xformer_8_10000.tar.gz"                 ##  for debugging only
# data_archive =  "en_es_xformer_8_90000.tar.gz"

max_seq_length = 10

embedding_size = 256               
#embedding_size = 128
#embedding_size = 64               

num_basic_encoders = num_basic_decoders = num_atten_heads = 4   
#num_basic_encoders = num_basic_decoders = num_atten_heads = 2  

masking = True                    ## For better results
#masking = False

dls = DLStudio(
                dataroot = dataroot,
                path_saved_model = {"encoder_PreLN" : "./saved_encoder_PreLN", 
                                    "decoder_PreLN" : "./saved_decoder_PreLN", 
                                    "embeddings_generator_en_PreLN" : "./saved_embeddings_generator_en_PreLN",
                                    "embeddings_generator_es_PreLN" : "./saved_embeddings_generator_es_PreLN",
                                   },
                learning_rate =  1e-5, 
                batch_size = 50,
                use_gpu = True,
                epochs = 60,
      )

xformer = TransformerPreLN( 
                            dl_studio = dls,
                            dataroot = dataroot,
                            save_checkpoints = True,
                            data_archive = data_archive,
                            max_seq_length = max_seq_length,
                            embedding_size = embedding_size,
          )

master_encoder = TransformerPreLN.MasterEncoder(
                                  dls,
                                  xformer,
                                  num_basic_encoders = num_basic_encoders,
                                  num_atten_heads = num_atten_heads,
                 )    


master_decoder = TransformerPreLN.MasterDecoderWithMasking(
                                  dls,
                                  xformer, 
                                  num_basic_decoders = num_basic_decoders,
                                  num_atten_heads = num_atten_heads,
                                  masking = masking,
                 )


number_of_learnable_params_in_encoder = sum(p.numel() for p in master_encoder.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Master Encoder: %d" % number_of_learnable_params_in_encoder)

number_of_learnable_params_in_decoder = sum(p.numel() for p in master_decoder.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the Master Decoder: %d" % number_of_learnable_params_in_decoder)

if masking:
    xformer.run_code_for_training_TransformerPreLN(dls,master_encoder,master_decoder,display_train_loss=True,
                                                                        checkpoints_dir="checkpoints_with_masking_PreLN")
else:
    xformer.run_code_for_training_TransformerPreLN(dls,master_encoder,master_decoder,display_train_loss=True,
                                                                        checkpoints_dir="checkpoints_no_masking_PreLN")

#import pymsgbox
#response = pymsgbox.confirm("Finished training.  Start evaluation?")
#if response == "OK": 

xformer.run_code_for_evaluating_TransformerPreLN(master_encoder, master_decoder)

def stop_word_removal(path_to_file):
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
    # print(lines)
    # remove EOS & SOS is the lines
    lines = [line.replace(' EOS', '') for line in lines]
    lines = [line.replace('SOS ', '') for line in lines]
    print(lines)

    with open(path_to_file, 'w') as f:
        f.writelines(lines)
    

path_to_FG = "translations_with_FG_40.txt"
path_to_PreLN = "translations_with_PreLN_60.txt"
stop_word_removal(path_to_FG)
stop_word_removal(path_to_PreLN)


def sort_txt_file(path_to_file):

    # sort the txt file into GT output pair
    with open(path_to_file, 'r') as f:
            input_data = f.readlines()

    output = ""
    for i in range(len(input_data)):
        save = 0
        
        # remove leading/trailing whitespace
        line = input_data[i].strip() 

        if line.startswith("The input sentence pair"):
            # split the line into parts by using the delimiter "['"
            parts = line.split("['")
            # print(parts)
            GoundT = parts[2].split("']")[0] # get the first part of the string
            # print(GT)
        elif line.startswith("The translation produced by"):
            parts = line.split(":")
            # print(parts)
            pred = parts[1].strip()
            # print(pred)
            save = 1

        if save == 1:
            output += GoundT + "\n" + pred + "\n"
            save = 0

    # remove the last empty line
    if output.endswith("\n"):
        output = output[:-1]
        
    return output



FG_output = sort_txt_file(path_to_FG)
with open("FG_output.txt", "w") as f:
        f.write(FG_output)
        
Pre_output = sort_txt_file(path_to_PreLN)
with open("Pre_output.txt", "w") as f:
        f.write(Pre_output)


import numpy as np

def levenshtein_distance(str1, str2):
    # Ensure str1 is the longer string
    if len(str1) < len(str2):
        str1, str2 = str2, str1

    len_str1, len_str2 = len(str1), len(str2)

    # Initialize two rows for dynamic programming
    previous_row = list(range(len_str2 + 1))
    current_row = [0] * (len_str2 + 1)

    for i in range(1, len_str1 + 1):
        current_row[0] = i
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            current_row[j] = min(
                previous_row[j] + 1,      # Deletion
                current_row[j - 1] + 1,   # Insertion
                previous_row[j - 1] + cost  # Substitution
            )
        previous_row, current_row = current_row, previous_row

    return previous_row[-1]




def report_statistics(distances):
    print(f"Mean: {np.mean(distances):.2f}")
    print(f"Median: {np.median(distances):.2f}")
    print(f"Standard Deviation: {np.std(distances):.2f}")
    print(f"Maximum: {np.max(distances):.2f}")
    print(f"Minimum: {np.min(distances):.2f}")

with open("FG_output.txt", "r") as f:
        FG_lines =f.readlines()
FG_distance = []
for i in range(0, len(FG_lines), 2):
    FG_distance.append(levenshtein_distance(FG_lines[i], FG_lines[i+1]))

print("TransformerFG Statistics:")
report_statistics(FG_distance)


with open("Pre_output.txt", "r") as f:
        Pre_lines =f.readlines()
Pre_distance = []
for i in range(0, len(Pre_lines), 2):
    Pre_distance.append(levenshtein_distance(Pre_lines[i], Pre_lines[i+1]))

print("TransformerPreLN Statistics:")
report_statistics(Pre_distance)
