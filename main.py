import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# initial config
CSV_COLUMN_NAMES = ['Phylums', 'Estructura ', 'Tejidos verdaderos', 'Simetria ', 'Cavidad Corporal con revestimiento',
                    'Sistema digestivo', 'Sistema digestivo2', 'Cuerpo']
PHYLUMS = ['Protozoarios', 'Poriferos', 'cnidarios', 'ctanoforos', 'platelmintos', 'nemertinos', 'asquelmintos',
           'acantocefalos', 'moluscos', 'anelidos', 'artropodos', 'equinodermos', 'cordados']

# set dataset path
train_path = 'phylum_train_dataset.csv'


# read data
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Phylums')
train_y = tf.keras.utils.to_categorical(train_y, num_classes=13)


# convert data to NumPy arrays
train_X = train.values



# define sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(train_X.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(13, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# train model
history = model.fit(train_X, train_y,batch_size = 4, shuffle=True, epochs=1000)
model.save('Phylum_recognition_model_V1.h5')





