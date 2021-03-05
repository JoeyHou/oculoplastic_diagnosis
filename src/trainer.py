import cv2
import dlib

import math
import numpy as np
import pandas as pd
from scipy import optimize
import seaborn as sns

import os
from tqdm import tqdm
import pickle

from utils import *
from data import *


class Trainer():
    def __init__(self, config):
        self.model_name = config['model_name'] + '_' + config['version']
        self.data_sources = config['data_sources']

        self.curr_dir = config['curr_dir']
        self.data_dir = self.curr_dir + config['data_dir']
        self.reference_dir = self.curr_dir + config['reference_dir']
        self.meta_data_dir = self.data_dir + 'meta_data/' + self.data_sources + '/'
        self.processed_dir = self.data_dir + 'processed/' + self.data_sources + '/'

        os.system('mkdir -p ' + self.data_dir)
        os.system('mkdir -p ' + self.data_dir + 'processed/' + self.data_sources)
        os.system('mkdir -p ' + self.data_dir + 'meta_data/' + self.data_sources)

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
        img_info_df = img_info_df.query('error == 0').drop(columns = ['landmark']).reset_index(drop = True)

        # Resizing images
        print('  => Now resizing images!')
        img_info_df['resized_img'] = [resize_to(512, self.processed_dir + fp) for fp in tqdm(img_info_df.filepath)]
        img_info_df['label'] = img_info_df.group.apply(lambda g: self.group_dict[g])

        pickle.dump(img_info_df, open(self.processed_dir + 'img_info_df_no_error.pickle', 'wb'))
        img_info_df.drop(columns = ['resized_img']).to_csv(self.meta_data_dir + 'img_info_df_no_error.csv', index = False)
        print(' => Done data_prep! Got', img_info_df.shape[0], 'images!')
        return 0
