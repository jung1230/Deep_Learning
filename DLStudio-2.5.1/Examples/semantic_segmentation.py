#!/usr/bin/env python

##  semantic_segmentation.py

"""
This script should be your starting point if you wish to learn how to use the
mUNet neural network for semantic segmentation of images.  As mentioned elsewhere in
the main documentation page, mUNet assigns an output channel to each different type of
object that you wish to segment out from an image. So, given a test image at the
input to the network, all you have to do is to examine each channel at the output for
segmenting out the objects that correspond to that output channel.
"""

import random
import numpy
import torch
import os, sys


"""
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
"""
if __name__ == "__main__":
## Add DLStudio-2.5.1 to sys.path so Python can find DLStudio
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n\ncurrent_dir = %s" % current_dir)
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    ##  watch -d -n 0.5 nvidia-smi

    from DLStudio import *

    dls = DLStudio(
    #                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5MultiObject/",
                      dataroot = "./../../data/datasets_for_DLStudio/data/",
                      image_size = [64,64],
                      path_saved_model = "./saved_model",
                      momentum = 0.9,
                      learning_rate = 1e-4,
                      epochs = 6,
                      batch_size = 4,
                      classes = ('rectangle','triangle','disk','oval','star'),
                      use_gpu = True,
                  )

    segmenter = DLStudio.SemanticSegmentation( 
                      dl_studio = dls, 
                      max_num_objects = 5,
                  )

    dataserver_train = DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                              train_or_test = 'train',
                              dl_studio = dls,
                              segmenter = segmenter,
                              dataset_file = "PurdueShapes5MultiObject-10000-train.gz", 
                            )
    dataserver_test = DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                              train_or_test = 'test',
                              dl_studio = dls,
                              segmenter = segmenter,
                              dataset_file = "PurdueShapes5MultiObject-1000-test.gz"
                            )
    segmenter.dataserver_train = dataserver_train
    segmenter.dataserver_test = dataserver_test

    segmenter.load_PurdueShapes5MultiObject_dataset(dataserver_train, dataserver_test)

    model = segmenter.mUNet(skip_connections=True, depth=16)
    #model = segmenter.mUNet(skip_connections=False, depth=4)

    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

    segmenter.run_code_for_training_for_semantic_segmentation(model)

    #import pymsgbox
    #response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
    #if response == "OK": 

    segmenter.run_code_for_testing_semantic_segmentation(model)

