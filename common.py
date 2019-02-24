import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Variables to control if we are using scaling or onehot encoding columns
scale_features_input = True
scale_features_output = True
use_categorical_feature_encoder = True

folder_characteristics = ''

if scale_features_input:
    folder_characteristics += '-inputsscaled'
else:
    folder_characteristics += '-inputsnotscaled'

if scale_features_output:
    folder_characteristics += '-outputsscaled'
else:
    folder_characteristics += '-outputsnotscaled'

if use_categorical_feature_encoder:
    folder_characteristics += '-categorical'
else:
    folder_characteristics += '-nocategorical'

folder_characteristics += '/'

# File Names
root_model_folder = './model/' + folder_characteristics

if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)

model_file_name = root_model_folder + 'model.h5'
model_checkpoint_file_name = root_model_folder + 'model-checkpoint.hdf5'

root_share_folder = './shared/'

# Column Names
X_colum_names = ['size', 'rooms', 'baths', 'parking', 'neighborhood']
Y_colum_names = ['price']
categorical_column = 'neighborhood'
