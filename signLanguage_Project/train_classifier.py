# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# import numpy as np

# import cv2 #for resizing the images

# data_dict = pickle.load(open('./data.pickle', 'rb')) #i'll probably move the pickle data to the sign language folder later

# # print(type(data_dict['data'][0]))
# # print(data_dict['data'][0])

# # data = [cv2.resize(d, (64, 64)) for d in data_dict['data']]
# data = np.array(data_dict['data'])
# labels = np.array(data_dict['labels'])


# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,shuffle=True ,stratify=labels)

# model = RandomForestClassifier()

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Fixing inconsistent lengths by padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# all sequences padded to the same length
data = pad_sequences(data_dict['data'], padding='post', dtype='float32')  # Pads with 0s
labels = np.array(data_dict['labels'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')  # Create a folder named 'saved_models'
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

model_path = os.path.join(save_dir, 'random_forest_model.pkl')  # File path for the model

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
