#!/usr/bin/env python
'''
This is a re-implementation of the following paper:
"Attention-based Deep Multiple Instance Learning"
I got very similar results but some data augmentation techniques not used here
https://128.84.21.199/pdf/1802.04712.pdf
*---- Jiawen Yao--------------*
'''

import os


import numpy as np
import time
from utl import Cell_Net
from random import shuffle
import argparse
from keras.models import Model
from utl.dataset import load_dataset
from utl.data_aug_op import random_flip_img, random_rotate_img
import glob
# import scipy.misc as sci this is depreciated
# from scipy.ndimage import imread 
import tensorflow as tf
import cv2
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from scipy.special import expit,softmax, logit
import matplotlib.pyplot as plt
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=1, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def generate_batch(path):
    bags = []
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.bmp')
        num_ins = len(img_path)

        label = int(each_path.split('/')[-2])

        if label == 1:
            curr_label = np.ones(num_ins,dtype=np.uint8)
        else:
            curr_label = np.zeros(num_ins, dtype=np.uint8)
        for each_img in img_path:
            img_data = np.asarray(cv2.imread(each_img), dtype=np.float32)
            #img_data -= 255
            img_data[:, :, 0] -= 123.68
            img_data[:, :, 1] -= 116.779
            img_data[:, :, 2] -= 103.939
            img_data /= 255
            # sci.imshow(img_data)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))

    return bags


def Get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    import random
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    

    num_test_batch = len(test_set)
    print('Size of test set batches') 
    print(num_test_batch)
    ak = len(test_set)
    ak = np.zeros((num_test_batch, 1), dtype=float)
    attention = K.function([model.layers[0].input], [model.layers[10].output])
    # np.save(attn_ak, ak)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    
    extract_conv1 = K.function([model.layers[0].input], [model.layers[1].output])
    # print('Extract conv1.shape')
    # print(extract_conv1)
    # Code for visuaization of a layer
    
    # load the image with the required shape
    img = load_img('sample_visualize.bmp', target_size=(27, 27))

    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)

    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)

    # get feature map for first hidden layer

    
    feature_maps = extract_conv1(img)

    # print('feature maps type ')
    # print(type(feature_maps))
    # for i in range(len(feature_maps)): 
    #     for x in feature_maps: 
    #         print(x[i], end =' ') 
    #     print() 

    print(' Feature map array')
    feature_map_array = np.array(feature_maps)
    print (feature_map_array.shape)
    print(feature_map_array)

    squeezed_feature_map_array = np.squeeze(feature_map_array)

    print(' squeezed feature map array')
    print(squeezed_feature_map_array.shape)
    (squeezed_feature_map_array)

    reshaped_squeezed_feature_map_array = squeezed_feature_map_array.reshape(1,24,24,36)

    # print('first feature map')
    # print(first_feature_map)
    #  feature_maps_array = np.array(feature_maps)
    # feature_maps_array = feature_maps_array/256
    # print('feature maps array')
    # print(type(feature_maps_array))

    # print('Feature maps array')
    # print(feature_maps_array) 
    # print(feature_maps_array.shape)
    # resized_feature_maps_array= feature_maps_array.reshape(24, 24,36)
    # print(resized_feature_maps_array.shape)
    
    square = 6 
    ix = 1
    for _ in range(square):
        for _ in range(square):
            #  specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(reshaped_squeezed_feature_map_array[0, :, :, ix-1], cmap='BuPu')
            ix += 1
    # show the figure
    plt.savefig('visualized_feature_maps.jpg')
    plt.show()

    for ibatch, batch in enumerate(test_set):
        result = model.test_on_batch(x=batch[0], y=batch[1])
        ak_k = batch[0]
        print('ak_k batch results of test on batch')
        print(result)

        print(' Printing zeroth element of the batch')
        # print(ak_k)
        print('ak_k. dimension')
        print(ak_k.ndim)
        print('ak_k. shape)')
        print(ak_k.shape)

        print('ak_k size')
        print(ak_k.size)

        print('And now the results of the attention function')
        print('Attention output')
        ak_output_list=attention(ak_k)
        # length_of_list = len(ak_output_list)
        ak_output_array = np.array(ak_output_list)

        print(' This is the ak output array')
        print (ak_output_array)
        
        # print('ak_output list is ')
        # print(ak_output_list)
        # print('ak_output_list first element is ')
        # print(ak_output_list[0])
        # print('Length of the list is ')
        # print(length_of_list)
        # print('ak_output_array. shape)')
        # print('ak_output_array. shape)')

        print('ak_output_array. shape)')

        print(ak_output_array.shape)
        print('ak_output_array. size)')

        print(ak_output_array.size)

        print('ak_output_array. dimension)')
        print(ak_output_array.ndim)

        print('Reshaped ak output array is')
        ak_output_array_size= ak_output_array.size

        reshaped_ak_outout_array = ak_output_array.reshape(ak_output_array_size)
        print('reshaped_ak_outout_array.shape)')
        print(reshaped_ak_outout_array.shape)

        print(reshaped_ak_outout_array.size)
        print('reshaped_ak_outout_array. size)')

        print('reshaped_ak_outout_array. dimension)')
        print(reshaped_ak_outout_array.ndim)

        ak_min = reshaped_ak_outout_array.min()
        ak_max = reshaped_ak_outout_array.max()
        scaled_ak_output = (reshaped_ak_outout_array - ak_min)/ (ak_max - ak_min)
        print('Scaled Attention weights')
        print(scaled_ak_output)
        
        y=softmax(reshaped_ak_outout_array)

        print('softmax output of  ak array')
        print(y)
        print('Softmax sum is ...')
        y_sum = np.sum(y)
        print(y_sum)
      

        #defining names of layers from which we will take the output
        layer_names = ['conv2d_1','max_pooling2d_1','conv2d_2','max_pooling2d_2']
        print('Shape of img')
        print(img.shape)
        outputs = []

        img1 = load_img('sample_visualize.bmp', target_size=(27, 27))

        img1 = img_to_array(img1)
        print('Shape of img1')
        print(img1.shape)

        # reshape data for the model
        img1 = img1.reshape((1, img1.shape[0], img1.shape[1], img1.shape[2]))
        # prepare the image for the VGG model

        """

        Keras works with batches of images. 
        So, the first dimension is used for the number of samples (or images) you have.
        When you load a single image, you get the shape of one image, which is (size1,size2,channels).
        In order to create a batch of images, you need an additional dimension: (samples, size1,size2,channels)
        The preprocess_input function is meant to adequate your image to the format the model requires.
        Some models use images with values ranging from 0 to 1. Others from -1 to +1
        """
        img1 = preprocess_input(img1)
        #extracting the output and appending to outputs
        print('Model .layers')


        for layer_name in layer_names:
            intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict(img1)
            outputs.append(intermediate_output)
            #plotting the outputs
        fig,ax = plt.subplots(nrows=4,ncols=5,figsize=(20,20))

        for i in range(4):
            for z in range(5):
                # Print the first five feature maps of each layer z=5
                ax[i][z].imshow(outputs[i][0,:,:,z])
                ax[i][z].set_title(layer_names[i])
                ax[i][z].set_xticks([])
                ax[i][z].set_yticks([])
        plt.savefig('layerwise_output.jpg')



        # ak_output_array = np.array(ak_output[0]).reshape((ak_k.shape[0]))
        # print('Ak output array')
        # print(ak_output_array)

        # np.savetxt('akoutput_save.txt', ak_output)
        # np.savetxt('akoutput1_save.txt', ak_output1)
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]
    return np.mean(test_loss), np.mean(test_acc)

def train_eval(model, train_set, irun, ifold):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set, train_percentage=0.9)

    from utl.DataGenerator import DataGenerator
    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    model_name = "Saved_model/" + "_Batch_size_" + str(batch_size) + "epoch_" + "best.hd5"

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    history = model.fit_generator(generator=train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             epochs=args.max_epoch, validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)


    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = 'Results/' + str(irun) + '_' + str(ifold) + "_val_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    return model_name


def model_training(input_dim, dataset, irun, ifold):

    train_bags = dataset['train']
    test_bags = dataset['test']

    # convert bag to batch
    train_set = generate_batch(train_bags)
    test_set = generate_batch(test_bags)

    model = Cell_Net.cell_net(input_dim, args, useMulGpu=False)

    # train model
    t1 = time.time()
    num_batch = len(train_set)
    # for epoch in range(args.max_epoch):
    model_name = train_eval(model, train_set, irun, ifold)

    print("load saved model weights")
    model.load_weights(model_name)

    test_loss, test_acc = test_eval(model, test_set)

    t2 = time.time()
    #

    print ('run time:', (t2 - t1) / 60.0, 'min')
    print ('test_acc={:.3f}'.format(test_acc))

    return test_acc



if __name__ == "__main__":

    args = parse_args()

    print ('Called with args:')
    print (args)

    input_dim = (27,27,3)

    run = 1
    n_folds = 10
    acc = np.zeros((run, n_folds), dtype=float)
    data_path = '/home/iiitb/amil/data/patches'


    for irun in range(run):
        dataset = load_dataset(dataset_path=data_path, n_folds=n_folds, rand_state=irun)
        for ifold in range(n_folds):
            print ('run=', irun, '  fold=', ifold)
            acc[irun][ifold] = model_training(input_dim, dataset[ifold], irun, ifold)
    print ('mi-net mean accuracy = ', np.mean(acc))
    print ('std = ', np.std(acc))

