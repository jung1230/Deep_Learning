# this code is mostly borrowed from DLStudio

import torch.optim as optim
import torch
import torch.nn as nn
import copy
import time
import gzip
import pickle
import numpy as np
import os
import sys
import torchvision
import matplotlib.pyplot as plt


class DLStudio(object):
    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DLStudio constructor can only be called with keyword arguments for 
                      the following keywords: epochs, learning_rate, batch_size, momentum,
                      convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                      image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                      debug_test''')
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        debug_train  = debug_test = None
        if 'dataroot' in kwargs                      :   dataroot = kwargs.pop('dataroot')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'convo_layers_config' in kwargs           :   convo_layers_config = kwargs.pop('convo_layers_config')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'fc_layers_config' in kwargs              :   fc_layers_config = kwargs.pop('fc_layers_config')
        if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes') 
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu') 
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train') 
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot:
            self.dataroot = dataroot
        if convo_layers_config:
            self.convo_layers_config = convo_layers_config
        if image_size:
            self.image_size = image_size
        if fc_layers_config:
            self.fc_layers_config = fc_layers_config
            if fc_layers_config[0] != -1:
                raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                                """The first element of the list of nodes in the fc layer must be -1 """
                                """because the input to fc will be set automatically to the size of """
                                """the final activation volume of the convolutional part of the network""")
        if  path_saved_model:
            self.path_saved_model = path_saved_model
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else: 
                    self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        self.debug_config = 0


    def imshow(self, img):
        '''
        called by display_tensor_as_image() for displaying the image
        '''
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()



    def display_tensor_as_image(self, tensor, title=""):
        '''
        This method converts the argument tensor into a photo image that you can display
        in your terminal screen. It can convert tensors of three different shapes
        into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
        number of pixels in the vertical direction and W, for width, for the same
        along the horizontal direction.  When the first element of the shape is 3,
        that means that the tensor represents a color image in which each pixel in
        the (H,W) plane has three values for the three color channels.  On the other
        hand, when the first element is 1, that stands for a tensor that will be
        shown as a grayscale image.  And when the shape is just (H,W), that is
        automatically taken to be for a grayscale image.
        '''
        tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
        if tensor_range == (-1.0,1.0):
            ##  The tensors must be between 0.0 and 1.0 for the display:
            print("\n\n\nimage un-normalization called")
            tensor = tensor/2.0 + 0.5     # unnormalize
        plt.figure(title)
        ###  The call to plt.imshow() shown below needs a numpy array. We must also
        ###  transpose the array so that the number of channels (the same thing as the
        ###  number of color planes) is in the last element.  For a tensor, it would be in
        ###  the first element.
        if tensor.shape[0] == 3 and len(tensor.shape) == 3:
#            plt.imshow( tensor.numpy().transpose(1,2,0) )
            plt.imshow( tensor.numpy().transpose(1,2,0) )
        ###  If the grayscale image was produced by calling torchvision.transform's
        ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
        ###  again have three elements in it, however the first element that stands for
        ###  the number of channels will now be 1
        elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
            tensor = tensor[0,:,:]
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        ###  For any one color channel extracted from the tensor representation of a color
        ###  image, the shape of the tensor will be (W,H):
        elif len(tensor.shape) == 2:
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        else:
            sys.exit("\n\n\nfrom 'display_tensor_as_image()': tensor for image is ill formed -- aborting")
        plt.show()




    class SemanticSegmentation(nn.Module):             
        """
        The purpose of this inner class is to be able to use the DLStudio platform for
        experiments with semantic segmentation.  At its simplest level, the purpose of
        semantic segmentation is to assign correct labels to the different objects in a
        scene, while localizing them at the same time.  At a more sophisticated level, a
        system that carries out semantic segmentation should also output a symbolic
        expression based on the objects found in the image and their spatial relationships
        with one another.

        The workhorse of this inner class is the mUNet network that is based on the UNET
        network that was first proposed by Ronneberger, Fischer and Brox in the paper
        "U-Net: Convolutional Networks for Biomedical Image Segmentation".  Their Unet
        extracts binary masks for the cell pixel blobs of interest in biomedical images.
        The output of their Unet can therefore be treated as a pixel-wise binary classifier
        at each pixel position.  The mUnet class, on the other hand, is intended for
        segmenting out multiple objects simultaneously form an image. [A weaker reason for
        "Multi" in the name of the class is that it uses skip connections not only across
        the two arms of the "U", but also also along the arms.  The skip connections in the
        original Unet are only between the two arms of the U.  In mUnet, each object type is
        assigned a separate channel in the output of the network.

        This version of DLStudio also comes with a new dataset, PurdueShapes5MultiObject,
        for experimenting with mUnet.  Each image in this dataset contains a random number
        of selections from five different shapes, with the shapes being randomly scaled,
        oriented, and located in each image.  The five different shapes are: rectangle,
        triangle, disk, oval, and star.

           Class Path:   DLStudio  ->  SemanticSegmentation
        """
        def __init__(self, dl_studio, max_num_objects, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.SemanticSegmentation, self).__init__()
            self.dl_studio = dl_studio
            self.max_num_objects = max_num_objects
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test


        class PurdueShapes5MultiObjectDataset(torch.utils.data.Dataset):
            """
            The very first thing to note is that the images in the dataset
            PurdueShapes5MultiObjectDataset are of size 64x64.  Each image has a random
            number (up to five) of the objects drawn from the following five shapes:
            rectangle, triangle, disk, oval, and star.  Each shape is randomized with
            respect to all its parameters, including those for its scale and location in the
            image.

            Each image in the dataset is represented by two data objects, one a list and the
            other a dictionary. The list data objects consists of the following items:

                [R, G, B, mask_array, mask_val_to_bbox_map]                                   ## (A)
            
            and the other data object is a dictionary that is set to:
            
                label_map = {'rectangle':50, 
                             'triangle' :100, 
                             'disk'     :150, 
                             'oval'     :200, 
                             'star'     :250}                                                 ## (B)
            
            Note that that second data object for each image is the same, as shown above.

            In the rest of this comment block, I'll explain in greater detail the elements
            of the list in line (A) above.

            
            R,G,B:
            ------

            Each of these is a 4096-element array whose elements store the corresponding
            color values at each of the 4096 pixels in a 64x64 image.  That is, R is a list
            of 4096 integers, each between 0 and 255, for the value of the red component of
            the color at each pixel. Similarly, for G and B.
            

            mask_array:
            ----------

            The fourth item in the list shown in line (A) above is for the mask which is a
            numpy array of shape:
            
                           (5, 64, 64)
            
            It is initialized by the command:
            
                 mask_array = np.zeros((5,64,64), dtype=np.uint8)
            
            In essence, the mask_array consists of five planes, each of size 64x64.  Each
            plane of the mask array represents an object type according to the following
            shape_index
            
                    shape_index = (label_map[shape] - 50) // 50
            
            where the label_map is as shown in line (B) above.  In other words, the
            shape_index values for the different shapes are:
            
                     rectangle:  0
                      triangle:  1
                          disk:  2
                          oval:  3
                          star:  4
            
            Therefore, the first layer (of index 0) of the mask is where the pixel values of
            50 are stored at all those pixels that belong to the rectangle shapes.
            Similarly, the second mask layer (of index 1) is where the pixel values of 100
            are stored at all those pixel coordinates that belong to the triangle shapes in
            an image; and so on.
            
            It is in the manner described above that we define five different masks for an
            image in the dataset.  Each mask is for a different shape and the pixel values
            at the nonzero pixels in each mask layer are keyed to the shapes also.
            
            A reader is likely to wonder as to the need for this redundancy in the dataset
            representation of the shapes in each image.  Such a reader is likely to ask: Why
            can't we just use the binary values 1s and 0s in each mask layer where the
            corresponding pixels are in the image?  Setting these mask values to 50, 100,
            etc., was done merely for convenience.  I went with the intuition that the
            learning needed for multi-object segmentation would become easier if each shape
            was represented by a different pixels value in the corresponding mask. So I went
            ahead incorporated that in the dataset generation program itself.

            The mask values for the shapes are not to be confused with the actual RGB values
            of the pixels that belong to the shapes. The RGB values at the pixels in a shape
            are randomly generated.  Yes, all the pixels in a shape instance in an image
            have the same RGB values (but that value has nothing to do with the values given
            to the mask pixels for that shape).
            
            
            mask_val_to_bbox_map:
            --------------------
                   
            The fifth item in the list in line (A) above is a dictionary that tells us what
            bounding-box rectangle to associate with each shape in the image.  To illustrate
            what this dictionary looks like, assume that an image contains only one
            rectangle and only one disk, the dictionary in this case will look like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25]], 
                                                150: [[37, 41, 55, 59]]}
            
            Should there happen to be two rectangles in the same image, the dictionary would
            then be like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25], [18, 16, 32, 36]], 
                                                150: [[37, 41, 55, 59]]}
            
            Therefore, it is not a problem even if all the objects in an image are of the
            same type.  Remember, the object that are selected for an image are shown
            randomly from the different shapes.  By the way, an entry like '[56, 20, 63,
            25]' for the bounding box means that the upper-left corner of the BBox for the
            'rectangle' shape is at (56,20) and the lower-right corner of the same is at the
            pixel coordinates (63,25).
            
            As far as the BBox quadruples are concerned, in the definition
            
                    [min_x,min_y,max_x,max_y]
            
            note that x is the horizontal coordinate, increasing to the right on your
            screen, and y is the vertical coordinate increasing downwards.

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  PurdueShapes5MultiObjectDataset

            """
            def __init__(self, dl_studio, segmenter, train_or_test, dataset_file):
                super(DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset, self).__init__()
                max_num_objects = segmenter.max_num_objects
                if train_or_test == 'train' and dataset_file == "PurdueShapes5MultiObject-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5MultiObject_label_map.pt"):
                        print("\nLoading training data from torch saved file")
                        self.dataset = torch.load("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        self.num_shapes = len(self.label_map)
                        self.image_size = dl_studio.image_size
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a few minutes.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.num_shapes = len(self.class_labels)
                        self.image_size = dl_studio.image_size
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.num_shapes = len(self.class_labels)
                    self.image_size = dl_studio.image_size

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image_size = self.image_size
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(image_size[0],image_size[1]), g.reshape(image_size[0],image_size[1]), b.reshape(image_size[0],image_size[1])
                im_tensor = torch.zeros(3,image_size[0],image_size[1], dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                mask_array = np.array(self.dataset[idx][3])
                max_num_objects = len( mask_array[0] ) 
                mask_tensor = torch.from_numpy(mask_array)
                mask_val_to_bbox_map =  self.dataset[idx][4]
                max_bboxes_per_entry_in_map = max([ len(mask_val_to_bbox_map[key]) for key in mask_val_to_bbox_map ])
                ##  The first arg 5 is for the number of bboxes we are going to need. If all the
                ##  shapes are exactly the same, you are going to need five different bbox'es.
                ##  The second arg is the index reserved for each shape in a single bbox
                bbox_tensor = torch.zeros(max_num_objects,self.num_shapes,4, dtype=torch.float)
                for bbox_idx in range(max_bboxes_per_entry_in_map):
                    for key in mask_val_to_bbox_map:
                        if len(mask_val_to_bbox_map[key]) == 1:
                            if bbox_idx == 0:
                                bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                        elif len(mask_val_to_bbox_map[key]) > 1 and bbox_idx < len(mask_val_to_bbox_map[key]):
                            bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                sample = {'image'        : im_tensor, 
                          'mask_tensor'  : mask_tensor,
                          'bbox_tensor'  : bbox_tensor }
                return sample

        def load_PurdueShapes5MultiObject_dataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)


        class SkipBlockDN(nn.Module):
            """
            This class for the skip connections in the downward leg of the "U"

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SkipBlockDN
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch

                # UP is using ConvTranspose2d, DN is using Conv2d
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if downsample:
                    # UP is using ConvTranspose2d, DN is using Conv2d
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out = out + identity
                    else:
                        out = out + torch.cat((identity, identity), dim=1) 
                return out


        class SkipBlockUP(nn.Module):
            """
            This class is for the skip connections in the upward leg of the "U"

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SkipBlockUP
            """
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockUP, self).__init__()
                self.upsample = upsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch

                # DN is using Conv2d, UP is using ConvTranspose2d
                self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)

                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if upsample:
                    # DN is using Conv2d, UP is using ConvTranspose2d
                    self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
            def forward(self, x):
                identity = x                                     
                out = self.convoT1(x)                              
                out = self.bn1(out)                              
                out = nn.functional.relu(out)
                out  =  nn.ReLU(inplace=False)(out)            
                if self.in_ch == self.out_ch:
                    out = self.convoT2(out)                              
                    out = self.bn2(out)                              
                    out = nn.functional.relu(out)
                if self.upsample:
                    out = self.upsampler(out)
                    identity = self.upsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out = out + identity                              
                    else:
                        out = out + identity[:,self.out_ch:,:,:]
                return out
            
        class ASPP(nn.Module):
            """
            This class is for the Atrous Spatial Pyramid Pooling (ASPP) block. The ASPP
            block is used to capture the context information at multiple scales.  The ASPP
            block uses atrous convolutions with different rates to capture context
            information at different scales.  The ASPP block is placed on top of the
            feature extractor network.  The ASPP block uses the original convolutional 
            feature map, the other branches use convolutional feature maps that are obtained by applying 
            atrous convolutions with different rates to the original feature map.  The outputs of
            the three branches are then concatenated and passed through a 1x1 convolutional
            layer to obtain the final output of the ASPP block.
            """
            def __init__(self, in_ch, out_ch):
                super(DLStudio.SemanticSegmentation.ASPP, self).__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

                # padding needs to be eqaul to dilation when kernel size is 3. So that the output size is 
                # same as input size. 
                # (formula: out_size = (in_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
                self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6) 
                self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
                self.conv4 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)
                self.conv5 = nn.Conv2d(in_ch, out_ch, 3, padding=24, dilation=24)

                # final conv for concatenation
                self.conv6 = nn.Conv2d(out_ch*5, out_ch, 1)
            def forward(self, x):
                out1 = self.conv1(x)
                out2 = self.conv2(x)
                out3 = self.conv3(x)
                out4 = self.conv4(x)
                out5 = self.conv5(x)
                out = torch.cat([out1, out2, out3, out4, out5], dim=1)
                out = self.conv6(out)
                return out
        

        class mUNet(nn.Module):
            """
            This network is called mUNet because it is intended for segmenting out
            multiple objects simultaneously form an image. [A weaker reason for "Multi" in
            the name of the class is that it uses skip connections not only across the two
            arms of the "U", but also also along the arms.]  The classic UNET was first
            proposed by Ronneberger, Fischer and Brox in the paper "U-Net: Convolutional
            Networks for Biomedical Image Segmentation".  Their UNET extracts binary masks
            for the cell pixel blobs of interest in biomedical images.  The output of their
            UNET therefore can therefore be treated as a pixel-wise binary classifier at
            each pixel position.

            The mUNet presented here, on the other hand, is meant specifically for
            simultaneously identifying and localizing multiple objects in a given image.
            Each object type is assigned a separate channel in the output of the network.

            I have created a dataset, PurdueShapes5MultiObject, for experimenting with
            mUNet.  Each image in this dataset contains a random number of selections from
            five different shapes, with the shapes being randomly scaled, oriented, and
            located in each image.  The five different shapes are: rectangle, triangle,
            disk, oval, and star.

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  mUNet

            """ 
            def __init__(self, skip_connections=True, depth=16):
                super(DLStudio.SemanticSegmentation.mUNet, self).__init__()
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
                ##  For the DN(down) arm of the U:
                self.bn1DN  = nn.BatchNorm2d(64)
                self.bn2DN  = nn.BatchNorm2d(128)
                self.skip64DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, skip_connections=skip_connections))
                self.skip64dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64,   downsample=True, skip_connections=skip_connections)
                self.skip64to128DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 128, skip_connections=skip_connections )
                self.skip128DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(128, 128, skip_connections=skip_connections))
                self.skip128dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)
                


                # add ASPP block before going up
                self.aspp = DLStudio.SemanticSegmentation.ASPP(128, 128)



                ##  For the UP arm of the U:
                self.bn1UP  = nn.BatchNorm2d(128)
                self.bn2UP  = nn.BatchNorm2d(64)
                self.skip64UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, skip_connections=skip_connections))
                self.skip64usUP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
                self.skip128to64UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 64, skip_connections=skip_connections )
                self.skip128UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(128, 128, skip_connections=skip_connections))
                self.skip128usUP = DLStudio.SemanticSegmentation.SkipBlockUP(128,128, upsample=True, skip_connections=skip_connections)
                self.conv_out = nn.ConvTranspose2d(64, 5, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                ##  Going down to the bottom of the U:
                x = nn.MaxPool2d(2,2)(nn.functional.relu(self.conv_in(x)))          
                for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                    x = skip64(x)                
        
                num_channels_to_save1 = x.shape[1] // 2 # x.shape[1] is the number of output channels
                save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                x = self.skip64dsDN(x)
                for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1DN(x)
                num_channels_to_save2 = x.shape[1] // 2
                save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                x = self.skip64to128DN(x)
                for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                    x = skip128(x)                
        
                x = self.bn2DN(x)
                num_channels_to_save3 = x.shape[1] // 2
                save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                    x = skip128(x)                
                x = self.skip128dsDN(x)



                # add ASPP block beforing going up
                x = self.aspp(x)



                ## Coming up from the bottom of U on the other side:
                x = self.skip128usUP(x)          
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                x = self.bn1UP(x)
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.skip128to64UP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                    x = skip64(x)                
                x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x = self.bn2UP(x)
                x = self.skip64usUP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                    x = skip64(x)                
                x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                x = self.conv_out(x)
                return x
        

        class SegmentationLoss(nn.Module):
            """
            I wrote this class before I switched to MSE loss.  I am leaving it here
            in case I need to get back to it in the future.  

            Class Path:   DLStudio  ->  SemanticSegmentation  ->  SegmentationLoss
            """
            def __init__(self, batch_size):
                super(DLStudio.SemanticSegmentation.SegmentationLoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, output, mask_tensor):
                composite_loss = torch.zeros(1,self.batch_size)
                mask_based_loss = torch.zeros(1,5)
                for idx in range(self.batch_size):
                    outputh = output[idx,0,:,:]
                    for mask_layer_idx in range(mask_tensor.shape[0]):
                        mask = mask_tensor[idx,mask_layer_idx,:,:]
                        element_wise = (outputh - mask)**2                   
                        mask_based_loss[0,mask_layer_idx] = torch.mean(element_wise)
                    composite_loss[0,idx] = torch.sum(mask_based_loss)
                return torch.sum(composite_loss) / self.batch_size


        def run_code_for_training_for_semantic_segmentation(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = criterion1(output, mask_tensor)  
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()    
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        avg_loss_segmentation = running_loss_segmentation / float(500)
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   MSE loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
            print("\nFinished Training\n")
            self.save_model(net)


        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_testing_semantic_segmentation(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            batch_size = self.dl_studio.batch_size
            image_size = self.dl_studio.image_size
            max_num_objects = self.max_num_objects
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    if i % 50 == 0:
                        print("\n\n\n\nShowing output for test batch %d: " % (i+1))
                        outputs = net(im_tensor)                        
                        ## In the statement below: 1st arg for batch items, 2nd for channels, 3rd and 4th for image size
                        output_bw_tensor = torch.zeros(batch_size,1,image_size[0],image_size[1], dtype=float)
                        for image_idx in range(batch_size):
                            for layer_idx in range(max_num_objects): 
                                for m in range(image_size[0]):
                                    for n in range(image_size[1]):
                                        output_bw_tensor[image_idx,0,m,n]  =  torch.max( outputs[image_idx,:,m,n] )
                        display_tensor = torch.zeros(7 * batch_size,3,image_size[0],image_size[1], dtype=float)
                        for idx in range(batch_size):
                            for bbox_idx in range(max_num_objects):   
                                bb_tensor = bbox_tensor[idx,bbox_idx]
                                for k in range(max_num_objects):
                                    i1 = int(bb_tensor[k][1])
                                    i2 = int(bb_tensor[k][3])
                                    j1 = int(bb_tensor[k][0])
                                    j2 = int(bb_tensor[k][2])
                                    output_bw_tensor[idx,0,i1:i2,j1] = 255
                                    output_bw_tensor[idx,0,i1:i2,j2] = 255
                                    output_bw_tensor[idx,0,i1,j1:j2] = 255
                                    output_bw_tensor[idx,0,i2,j1:j2] = 255
                                    im_tensor[idx,0,i1:i2,j1] = 255
                                    im_tensor[idx,0,i1:i2,j2] = 255
                                    im_tensor[idx,0,i1,j1:j2] = 255
                                    im_tensor[idx,0,i2,j1:j2] = 255
                        display_tensor[:batch_size,:,:,:] = output_bw_tensor
                        display_tensor[batch_size:2*batch_size,:,:,:] = im_tensor

                        for batch_im_idx in range(batch_size):
                            for mask_layer_idx in range(max_num_objects):
                                for i in range(image_size[0]):
                                    for j in range(image_size[1]):
                                        if mask_layer_idx == 0:
                                            if 25 < outputs[batch_im_idx,mask_layer_idx,i,j] < 85:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 1:
                                            if 65 < outputs[batch_im_idx,mask_layer_idx,i,j] < 135:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 2:
                                            if 115 < outputs[batch_im_idx,mask_layer_idx,i,j] < 185:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 3:
                                            if 165 < outputs[batch_im_idx,mask_layer_idx,i,j] < 230:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 4:
                                            if outputs[batch_im_idx,mask_layer_idx,i,j] > 210:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[2*batch_size+batch_size*mask_layer_idx+batch_im_idx,:,:,:]= outputs[batch_im_idx,mask_layer_idx,:,:]
                        self.dl_studio.display_tensor_as_image(
                           torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))



if __name__ == '__main__':

    dls = DLStudio(
    #                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5MultiObject/",
                        dataroot = "./../data/datasets_for_DLStudio/data/",
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

    # segmenter.run_code_for_training_for_semantic_segmentation(model)

    print("\nFinished training.")


    segmenter.run_code_for_testing_semantic_segmentation(model)