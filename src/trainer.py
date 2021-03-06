import cv2
import dlib

import math
import numpy as np
import pandas as pd
from scipy import optimize
import seaborn as sns
from datetime import date

import os
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch.optim as optim

# import torchvision.transforms as transforms
# from torchvision import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from utils import *
from data import *
from model import *

class Trainer():
    def __init__(self, config):
        print()
        print(' => In Trainer.__init__()!')

        # Runtime info
        self.model_name = config['model_name'] + '_' + config['version']
        self.curr_dir = config['curr_dir']

        # Data dir
        self.reference_dir = self.curr_dir + config['reference_dir']
        self.data_dir = self.curr_dir + config['data_dir']
        os.system('mkdir -p ' + self.data_dir)

        # Image data dir
        self.data_sources = config['data_sources']
        self.processed_dir = self.data_dir + 'processed/' + self.data_sources + '/'
        os.system('mkdir -p ' + self.processed_dir)

        # Meta data dir
        self.meta_data_dir = self.data_dir + 'meta_data/' + self.data_sources + '/'
        os.system('mkdir -p ' + self.meta_data_dir)

        # Models dir
        self.model_dir = self.data_dir + 'models/' + self.data_sources + '/'
        os.system('mkdir -p ' + self.model_dir)
        os.system('mkdir -p ' + self.curr_dir + 'tmp/')

        # Loggings
        self.training_log = self.meta_data_dir + self.data_sources + 'training_log.txt'
        self.testing_log = self.meta_data_dir + self.data_sources + 'testing_log.txt'
        self.training_log_lst = []

        # Model info
        self.model_config = config['model_config']
        self.original_size = (256, 512) # self.model_config['original_size']
        self.batch_size = self.model_config['batch_size']
        self.curr_label = self.model_config['curr_label']
        self.class_num = self.model_config['class_num']
        self.n_epochs = self.model_config['n_epochs']
        self.from_checkpoint = self.model_config['from_checkpoint']

        # Group info
        self.group_dict = {}
        self.group_lst = []
        # Check if raw data is available
        if self.data_sources not in os.listdir(self.data_dir + 'raw/'):
            print('[ERROR] Data source:', self.data_sources, 'not found in raw data directory! STOP!')
            return 1
        else:
            group_counter = 0
            for i in os.listdir(self.data_dir + 'raw/' + self.data_sources):
                if os.path.isdir(self.data_dir + 'raw/' + self.data_sources + '/' + i):
                    self.group_lst.append(i)
                    self.group_dict[i] = group_counter
                    group_counter += 1
            print('  => Found raw data source with these groups:', self.group_lst, ', with group directory:', self.group_dict)
        print(' => Done init!')

    def clean_group(self, group_name, landmark_predictor, face_detector):
        counter = 0
        raw_file_dir = self.data_dir + 'raw/' + self.data_sources + '/'
        # processed_dir = self.data_dir + 'processed/' + self.data_sources + '/'
        os.system('mkdir -p ' + self.processed_dir + group_name)
        print("  => Cleaning group:", group_name)

        error_img = []
        img_info_dic = []
        # print(listdir(raw_file_dir + group_name + '/'))
        for f in tqdm(listdir(raw_file_dir + group_name + '/')):
            if '.JPG' not in f:
                continue

            output, _, shape = predict_landmarks(raw_file_dir + group_name + '/' + f, landmark_predictor, face_detector)
            shape = shape[36:48, :]

            new_key = str(counter)
            new_name = group_name + '_' + new_key + '.JPG'
            tmp_dic = {}
            tmp_dic['original_name'] = f
            tmp_dic['new_name'] = new_name
            tmp_dic['group'] = group_name

            # Cropping
            inter_w = shape[6][0] - shape[3][0]
            left = int(shape[0][0] - inter_w / 2)
            right = int(shape[9][0] + inter_w / 2)
            up = min(shape[:, 1])
            down = max(shape[:, 1])
            inter_h = down - up
            up = up - inter_h
            down = down + inter_h
            output = output[up: down, left: right]
            shape = shape - np.array([left, up])
            tmp_dic['landmark'] = [tuple(i) for i in shape]

            # Saving
            if 0 in output.shape:
                error_img.append(tmp_dic)
                continue
            img_info_dic.append(tmp_dic)
            cv2.imwrite(self.processed_dir + group_name + '/' + new_name, output)

            counter += 1
            # if counter > 2:
            #     break
        return img_info_dic, error_img

    def data_cleaning(self):
        print()
        print(' => In data_cleaning!')

        # Predictor file path info
        predictor_path = self.reference_dir + 'shape_predictor_68_face_landmarks.dat'
        landmark_predictor = dlib.shape_predictor(predictor_path)
        face_detector = dlib.get_frontal_face_detector()

        # Clean each group
        img_info_dic = []
        error_img = []
        for group in self.group_lst:
            tmp_info_dic, tmp_error_img = self.clean_group(group, landmark_predictor, face_detector) # in data.py
            print(tmp_info_dic, tmp_error_img)
            print()
            print()
            img_info_dic += tmp_info_dic
            error_img += tmp_error_img
        img_info_df = pd.DataFrame(img_info_dic)
        error_df = pd.DataFrame(error_img)
        img_info_df.to_csv(self.meta_data_dir + 'img_info_df.csv', index = False)
        error_df.to_csv(self.meta_data_dir + 'error_img_df.csv', index = False)

        print(' => Done data_cleaning! Got', img_info_df.shape[0], 'images!')


    def data_prep(self):
        print()
        print(' => In data_prep!')
        # print(self.group_dict)

        img_info_df = pd.read_csv(self.meta_data_dir + 'img_info_df.csv')
        error_img_df = pd.read_csv(self.meta_data_dir + 'error_img_df.csv')
        error_img_names = set(error_img_df.new_name.values)

        # Construct the needed columns
        img_info_df['error'] = img_info_df.new_name.apply(lambda n: 1 if n in error_img_names else 0)
        img_info_df['filepath'] = img_info_df.apply(lambda s: s.group + '/' + s.new_name, axis = 1)

        num_error = img_info_df.query('error == 1').shape[0]
        num_normal = img_info_df.query('group == "normal" & error == "0"').shape[0]
        print('  => Out of', img_info_df.shape[0], 'images,', num_error, 'were not cleaned/cropped correctly.')
        print('  => Out of', img_info_df.shape[0], 'images,', num_normal, 'were cleaned/cropped correctly AND had a label of "normal eyes".')

        # Drop un-necessary columns
        img_info_df_no_error = img_info_df.query('error == 0').drop(columns = ['landmark']).reset_index(drop = True)

        # Resizing images
        print('  => Now resizing images!')
        img_info_df_no_error['resized_img'] = [resize_to(512, self.processed_dir + fp) for fp in tqdm(img_info_df_no_error.filepath)]
        img_info_df_no_error['label'] = img_info_df_no_error.group.apply(lambda g: self.group_dict[g])

        # Assigining train-val-test
        all_idx = list(range(img_info_df_no_error.shape[0]))
        img_info_df_no_error['img_idx'] = all_idx
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(all_idx, all_idx, test_size = 0.3, random_state = 42)
        X_val_idx, X_test_idx, y_val, y_test = train_test_split(X_test_idx, y_test, test_size = 0.5, random_state = 42)
        # print(len(X_train_idx), len(X_val_idx), len(X_test_idx))
        # img_info_df_no_error['dataset'] = all_idx
        def calc_dataset(s):
            if s in X_train_idx:
                return 'train'
            if s in X_val_idx:
                return 'val'
            if s in X_test_idx:
                return 'test'
        img_info_df_no_error['dataset'] = [calc_dataset(idx) for idx in tqdm(img_info_df_no_error.img_idx.values)]

        # Adding customized labels
        for group in self.group_lst:
            img_info_df_no_error['label_' + group] = img_info_df_no_error.group.apply(lambda g: 1 if g == group else 0)

        # Saving data to pickle file
        pickle.dump(img_info_df_no_error, open(self.processed_dir + 'img_info_df_no_error.pickle', 'wb'))
        img_info_df_no_error.drop(columns = ['resized_img']).to_csv(self.meta_data_dir + 'img_info_df_no_error.csv', index = False)
        print(' => Done data_prep! Got', img_info_df_no_error.shape[0], 'images!')
        return 0

    def dataloader_helper(self, df, train = False):
        img_vec = torch.tensor([i for i in df.resized_img.values])
        img_vec = img_vec.reshape((img_vec.shape[0], 1, img_vec.shape[1], img_vec.shape[2])).float()
        data_set = TensorDataset(img_vec,
                                 torch.tensor(df[self.curr_label].values),
                                 torch.tensor(df.img_idx.values))
        if train:
            return DataLoader(data_set, sampler = RandomSampler(data_set), batch_size = self.batch_size)
        else:
            return DataLoader(data_set, sampler = SequentialSampler(data_set), batch_size = self.batch_size)

    def dataloader_prep(self):
        print('  => In dataloader_prep!')

        # Load pickle file
        # data_file = open(self.data_pkl_dir, 'rb')
        data = pd.read_pickle(self.processed_dir + 'img_info_df_no_error.pickle')

        # Make dataloaders
        train_df = data.query('dataset == "train"').reset_index(drop = True)
        val_df = data.query('dataset == "val"').reset_index(drop = True)
        test_df = data.query('dataset == "test"').reset_index(drop = True)

        train_dataloader = self.dataloader_helper(train_df, train = True)
        val_dataloader = self.dataloader_helper(val_df)
        test_dataloader = self.dataloader_helper(test_df)

        print('  => Done dataloader_prep!')
        return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        print()
        print(' => Training!')
        # Data preparation
        train_dataloader, val_dataloader, test_dataloader = self.dataloader_prep()

        # create a complete CNN
        self.model_config['original_size'] = self.original_size
        model = DiagnoisisNet(self.model_config)
        # print(model)
        # move tensors to GPU if CUDA is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('  => Using cuda :)')
            model.cuda()
        else:
            print('  => No GPU :/')
        # specify loss function (categorical cross-entropy)
        criterion = nn.CrossEntropyLoss()
        # specify optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        # optimizer = optim.SGD(model.parameters(), lr=0.00001)

        valid_loss_min = np.Inf # track change in validation loss

        starting_epoch = self.from_checkpoint + 1
        best_test_acc = 0
        for epoch in tqdm(range(starting_epoch, starting_epoch + self.n_epochs)):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            model.train()

            # indices = torch.randperm(len(X_train), dtype = torch.long, device = 'cpu')
            # train_dataloader = torch.utils.data.DataLoader(indices, batch_size = batch_size)
            train_pred = []
            train_label = []
            train_scores = []
            for batch in train_dataloader:
                data, target, img_names = batch
                # print(data.shape, target.shape, img_names.shape)
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # print(output.shape)
                # print(output.shape, target.shape)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                if train_on_gpu:
                    output_scores = output.to('cpu').detach().numpy()
                else:
                    output_scores = output.detach().numpy()
                output = np.argmax(output_scores, axis = 1).flatten()
                for o in output:
                    train_pred.append(o)
                for s in output_scores:
                    train_scores.append(s)
                for t in target:
                    train_label.append(t)

            train_acc = sum(np.array(train_pred) == np.array(train_label)) / len(train_pred)
            # train_acc = output_analysis(np.array(train_pred), np.array(train_label))['acc']

            ######################
            # validate the model #
            ######################
            model.eval()
            # indices = list(range(len(X_val)))
            # test_dataloader = torch.utils.data.DataLoader(indices, batch_size = batch_size)
            test_pred = []
            test_label = []
            test_scores = []
            for batch in test_dataloader:
                data, target, img_names = batch
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                with torch.no_grad():
                    output = model(data)
                # calculate the batch loss

                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)
                # Update for acc
                if train_on_gpu:
                    output_scores = output.to('cpu').detach().numpy()
                else:
                    output_scores = output.detach().numpy()
                output = np.argmax(output_scores, axis = 1).flatten()
                for o in output:
                    test_pred.append(o)
                for s in output_scores:
                    test_scores.append(s)
                for t in target:
                    test_label.append(t)
            # test_acc = output_analysis(np.array(test_pred), np.array(test_label))['acc']
            test_acc = sum(np.array(test_pred) == np.array(test_label)) / len(test_pred)

            # calculate average losses
            train_loss = train_loss/len(train_dataloader)
            valid_loss = valid_loss/len(test_dataloader)

            # if epoch % 10 == 0:
            #     # print training/validation statistics
            #     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            #         epoch, train_loss, valid_loss))
            #     print('Train acc:', train_acc, '; vali acc:', test_acc)

            # save model if validation loss has decreased
            if test_acc > best_test_acc:
                torch.save(model.state_dict(), self.curr_dir + 'tmp/tmp_model.pt')
                best_test_acc = test_acc
            tmp_summary_dict = {}
            tmp_summary_dict['epoch'] = epoch
            tmp_summary_dict['val_acc'] = test_acc
            tmp_summary_dict['train_acc'] = train_acc
            tmp_summary_dict['train_loss'] = train_loss
            tmp_summary_dict['valid_loss'] = valid_loss
            self.training_log_lst.append(tmp_summary_dict)
        training_summary_df = pd.DataFrame(self.training_log_lst)
        training_summary_df.to_csv(self.meta_data_dir + 'training_log.csv', index = False)

        model = DiagnoisisNet(self.model_config)
        model.load_state_dict(torch.load(self.curr_dir + 'tmp/tmp_model.pt'))
        test_acc = self.run_single_test(model, test_dataloader, return_prediction_dict = False)
        print(' => Final test acc:', test_acc)
        torch.save(model.state_dict(), self.model_dir + 'model_' + self.model_name + '.pt')
        return training_summary_df, test_acc

    def run_single_test(self, model, dataloader, return_prediction_dict = True):
        print()
        print(" => Testing!")
        test_on_gpu = torch.cuda.is_available()
        if test_on_gpu:
            print('  => Testing on cuda :)')
            model.cuda()
        else:
            print('  => No GPU :/')
        # model.cuda()

        model.eval()
        nb_eval_steps, nb_eval_examples = 0, 0
        test_label = []
        test_pred = []
        test_scores = []
        all_prediction_dict = {}
        first_batch = True
        # Evaluate data for one epoch
        for batch in tqdm(dataloader):

            data, target, img_names = batch
             # move tensors to GPU if CUDA is available
            if test_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            with torch.no_grad():
                output = model(data)

            # loss = criterion(output, target)
            # # update average validation loss
            # valid_loss += loss.item()*data.size(0)
            # Update for acc
            if test_on_gpu:
                output_scores = output.to('cpu').detach().numpy()
            else:
                output_scores = output.detach().numpy()
            output = np.argmax(output_scores, axis = 1).flatten()
            for o in output:
                test_pred.append(o)
            for s in output_scores:
                test_scores.append(s)
            for t in target:
                test_label.append(t)

            for i in range(len(img_names)):
                all_prediction_dict[img_names[i]] = [output_scores[i], output[i]]
        # test_acc = output_analysis(np.array(test_pred), np.array(test_label))['acc']
        test_acc = sum(np.array(test_pred) == np.array(test_label)) / len(test_pred)
        # print(test_acc)
        if return_prediction_dict:
            return test_acc, all_prediction_dict
        else:
            return test_acc
