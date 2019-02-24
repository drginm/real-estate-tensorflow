import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import common

# Get Categorical Feature Encoder
label_encoder_file_name = common.root_model_folder + 'label_encoder.sav'
onehot_encoder_file_name = common.root_model_folder + 'onehot_encoder.sav'

def create_categorical_feature_encoder(column):
    label_encoder = LabelEncoder()
    label_encoder.fit(column)

    onehot_encoder = OneHotEncoder(sparse=False)

    neighborhood_column = label_encoder.transform(column)
    integer_encoded = neighborhood_column.reshape(len(neighborhood_column), 1)
    onehot_encoder.fit(integer_encoded)

    # Save encoders state for later use
    pickle.dump(label_encoder, open(label_encoder_file_name, 'wb'))
    pickle.dump(onehot_encoder, open(onehot_encoder_file_name, 'wb'))

    return label_encoder, onehot_encoder

def load_categorical_feature_encoder():
    label_encoder = pickle.load(open(label_encoder_file_name, 'rb'))
    onehot_encoder = pickle.load(open(onehot_encoder_file_name, 'rb'))

    return label_encoder, onehot_encoder

def encode_categorical_column(dataset, column_name, label_encoder, onehot_encoder):
    dataset[column_name] = label_encoder.transform(dataset[column_name])
    integer_encoded = dataset[column_name].values.reshape(len(dataset[column_name]), 1)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    dataset = dataset.reset_index(drop=True)
    dataset = pd.concat([dataset, pd.DataFrame(onehot_encoded)], axis=1)

    dataset = dataset.drop([column_name], axis=1)
    return dataset
