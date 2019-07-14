from sklearn.preprocessing import OneHotEncoder, LabelEncoder

neighborhoods = [
    'envigado',
    'poblado',
    'centro',
    'laureles',
    'bello',
]

labels_to_test = [
    'laureles',
    'centro',
    'laureles',
]

label_encoder = LabelEncoder()
label_encoder.fit(neighborhoods)

print('Label Encoded String', label_encoder.transform(labels_to_test))
