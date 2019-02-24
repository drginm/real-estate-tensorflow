import pandas as pd

import common
import common_scaler
import common_categorical

#Transform everything!
def transform_inputs(values, label_encoder, onehot_encoder, x_scaler, y_source = None, y_scaler = None):
    values = pd.DataFrame(values, columns=common.X_colum_names)

    if common.use_categorical_feature_encoder:
        values_df = common_categorical.encode_categorical_column(values, common.categorical_column, label_encoder, onehot_encoder)
    else:
        values_df = values
        values_df[common.categorical_column] = label_encoder.transform(values_df[common.categorical_column])

    if common.scale_features_input:
        arr_x_predict, arr_y_predict = common_scaler.scale_dataset(values_df, x_scaler, y_source, y_scaler)
    else:
        arr_x_predict, arr_y_predict = values_df.values, y_source

    return arr_x_predict, arr_y_predict

def transform_outputs(y_predicted, y_scaler):
    if common.scale_features_output:
        y_predicted_unscaled = y_scaler.inverse_transform(y_predicted)
    else:
        y_predicted_unscaled = y_predicted

    return y_predicted_unscaled
