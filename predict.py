import numpy as np
import tensorflow as tf
import pandas as pd

PHYLUMS = ['Protozoarios', 'Poriferos', 'cnidarios', 'ctanoforos', 'platelmintos', 'nemertinos', 'asquelmintos',
           'acantocefalos', 'moluscos', 'anelidos', 'artropodos', 'equinodermos', 'cordados']

model = tf.keras.models.load_model("Phylum_recognition_model_V1.h5")
input_data = np.array([1,0,0.66,0,0,0,1])
input_data = input_data.reshape(1, -1)
prediction = model.predict(input_data)
phylumIndex = np.argmax(prediction)
print(PHYLUMS[phylumIndex])
print(np.max(prediction))
