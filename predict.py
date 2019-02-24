#%%
from tensorflow.keras.models import load_model

import common
import common_scaler
import common_categorical
import common_pre_post_processing

# Load the previous state of the model
model = load_model(common.model_file_name)

# Load the previous state of the enconders and scalers
label_encoder, onehot_encoder = common_categorical.load_categorical_feature_encoder()
x_scaler, y_scaler = common_scaler.load_scaler()

#Some inputs to predict
# 'size', 'rooms', 'baths', 'parking', 'neighborhood'
values = [
    [180, 5, 2, 0, 'envigado'],
    [180, 5, 2, 0, 'medellin belen'],
    [180, 5, 2, 0, 'sabaneta zaratoga'],
    #310000000,97,3,2,2,sabaneta centro
    [ 97, 3, 2, 2, 'sabaneta centro'],
    #258000000,105,3,2,0,medellin belen
    [105, 3, 2, 0, 'medellin belen'],
    #335000000,160,3,3,2,medellin la mota
    [160, 3, 3, 2, 'medellin la mota'],
]

# Transform inputs to the format that the model expects
model_inputs, _ = common_pre_post_processing.transform_inputs(values, label_encoder, onehot_encoder, x_scaler)

# Use the model to predict the price for a house
y_predicted = model.predict(model_inputs)

# Transform the results into a user friendly representation
y_predicted_unscaled = common_pre_post_processing.transform_outputs(y_predicted, y_scaler)

print('Results when:')
print('Scale Input Features = ', common.scale_features_input)
print('Scale Output Features = ', common.scale_features_output)
print('Use Categorical Feature Eencoder  = ', common.use_categorical_feature_encoder)

for i in range(0, len(values)):
    print(values[i][4], y_predicted[i][0], int(y_predicted_unscaled[i]))
