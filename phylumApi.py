from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf

PHYLUMS = ['Protozoarios', 'Poriferos', 'cnidarios', 'ctanoforos', 'platelmintos', 'nemertinos', 'asquelmintos',
           'acantocefalos', 'moluscos', 'anelidos', 'artropodos', 'equinodermos', 'cordados']

model = tf.keras.models.load_model("Phylum_recognition_model_V1.h5")

app = Flask(__name__)

@app.get('/healthCheck')
def healthCheck():
    return 'Ok'

@app.post('/phylum')
def getPhylum():
    data = request.get_json()
    answers = data['answers']
    print(data)
    if not answers or len(answers) != 7 :
        return jsonify({"error": "Wrong data format"}), 400

    input_data = np.array(answers)
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    phylumIndex = np.argmax(prediction)
    phylumPredicted = PHYLUMS[phylumIndex]
    # Respuesta
    return jsonify({
        "prediction": phylumPredicted,
        "accuracyPercentage": str(round(np.max(prediction),4))
    })

if __name__ == "__main__":
    app.run(debug = True, port = 3333)