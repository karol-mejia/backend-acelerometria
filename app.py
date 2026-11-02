from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# Modelo de ejemplo con Random Forest
X_dummy = np.random.rand(100, 6)
y_dummy = np.random.choice(['sentada', 'acostada', 'marcha', 'bipeda'], 100)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_dummy, y_dummy)

@app.route('/analizar', methods=['POST'])
def analizar_acelerometria():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo'}), 400

    file = request.files['file']
    filename = file.filename

    # Lectura del archivo
    if filename.endswith('.csv'):
        df = pd.read_csv(file, skiprows=1, sep=';', decimal=',')
    else:
        df = pd.read_csv(file, skiprows=14, sep=r'\s+', decimal=',')

    # Extracción de características para el Random Forest
    features = [[
        df.iloc[:, 1].mean(), df.iloc[:, 1].std(),
        df.iloc[:, 2].mean(), df.iloc[:, 2].std(),
        df.iloc[:, 3].mean(), df.iloc[:, 3].std()
    ]]

    actividad_predicha = modelo_rf.predict(features)[0]

    return jsonify({
        'paciente': filename.replace('.csv', '').replace('.txt', ''),
        'actividad': actividad_predicha,
        'registrosTotales': len(df),
        'tiempoTotal': f"{len(df)/100.0:.1f} s"
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
