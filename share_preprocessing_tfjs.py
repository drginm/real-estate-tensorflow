#%%
import common
import common_file
import common_scaler
import common_categorical

#%% Save values to be shared with the web app
# Load the previous state of the enconders
label_encoder, onehot_encoder = common_categorical.load_categorical_feature_encoder()

enconder_classes = list(label_encoder.classes_)

common_file.generate_json_file(enconder_classes, common.root_share_folder, 'neighborhoods')

print(enconder_classes)

#%% Save values to be shared with the web app
# Load the previous state of the scalers
x_scaler, y_scaler = common_scaler.load_scaler()

mean_x = x_scaler.mean_
var_x = x_scaler.var_

common_file.generate_json_file(list(mean_x), common.root_share_folder, 'scaler-mean-x')
common_file.generate_json_file(list(var_x), common.root_share_folder, 'scaler-var-x')

mean_y = y_scaler.mean_
var_y = y_scaler.var_

common_file.generate_json_file(list(mean_y), common.root_share_folder, 'scaler-mean-y')
common_file.generate_json_file(list(var_y), common.root_share_folder, 'scaler-var-y')

print('X values')
print(mean_x)
print(var_x)
print('Y values')
print(mean_y)
print(var_y)
