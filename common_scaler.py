import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

import common

# Get Scaler
x_scaler_file_name = common.root_model_folder + 'x_scaler.sav'
y_scaler_file_name = common.root_model_folder + 'y_scaler.sav'

def create_scaler(x_values, y_values):
    x_scaler = StandardScaler()
    x_scaler.fit(x_values)

    y_scaler = StandardScaler()
    y_scaler.fit(y_values)

    # Save scalers state for later use
    pickle.dump(x_scaler, open(x_scaler_file_name, 'wb'))
    pickle.dump(y_scaler, open(y_scaler_file_name, 'wb'))

    return x_scaler, y_scaler

def load_scaler():
    x_scaler = pickle.load(open(x_scaler_file_name, 'rb'))
    y_scaler = pickle.load(open(y_scaler_file_name, 'rb'))

    return x_scaler, y_scaler

def scale_dataset(x_source, x_scaler, y_source = None, y_scaler = None):
    scaled_columns = x_scaler.transform(x_source.values[:,0:4])
    arr_x_train = np.concatenate(( scaled_columns, x_source.values[:,5:]), axis=1)

    if y_source is not None:
        arr_y_train = y_scaler.transform(y_source.values)
    else:
        arr_y_train = None

    return arr_x_train, arr_y_train
