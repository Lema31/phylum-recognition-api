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


# convert data to NumPy arrays
train_X = train.values
train_y = train_y.values


# define sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation='relu', input_shape=(train_X.shape[1],)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(13, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train model
history = model.fit(train_X, train_y,validation_split = 0.2, shuffle=True, epochs=1000)
model.save('Phylum_recognition_model_V1.h5')

# evaluate model on test data

plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


