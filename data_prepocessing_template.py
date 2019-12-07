import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#MISSING DATA
from sklearn.prepocessing import Imputer
imputer = Imputer(missing_values = 'Nan', strategy = 'mean', axis = '0')
imputer = imputer.fit(dataset)
dataset = imputer.transform(dataset) # NEED TO CHECK IF WE CAN USE fit_transform

# ONE HOT ENCODING - first convert strings to number and then numbers to binary
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_features = LabelEncoder()
new_feats_with_lab_cats = labelencoder_features.fit_transform(categorical_part_of_feats)

onehotencoder = OneHotEncoder(categorical_features = [new_cat_feats_column_numbers])
one_hot_encoded_feats = onehotencoder.fit_transform(new_feats_with_lab_cats).toarray()

label_encoder_targets = LabelEncoder()
targets = labelencoder_y.fit_transform(targets)


# SPLIT TRAIN AND TEST
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(    X_test )
