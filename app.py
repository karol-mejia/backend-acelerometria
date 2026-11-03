import os
import re
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'modelo_random_forest.pkl'
if os.path.exists(MODEL_PATH):
    modelo_rf = joblib.load(MODEL_PATH)
    print("¡Modelo Random Forest cargado exitosamente!")
else:
    modelo_rf = None
    print("Advertencia: No se encontró el archivo del modelo .pkl en el repositorio.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "mensaje": "¡El backend de acelerometría está funcionando correctamente!"
    })

def extraer_caracteristicas_ventana(df_acelerometria):
    ventana_tamanio = 100
    caracteristicas = []
    
    for start in range(0, len(df_acelerometria), ventana_tamanio):
        window = df_acelerometria.iloc[start:start+ventana_tamanio]
        if len(window) < ventana_tamanio:
            break
            
        feat = {
            'mean_frontal': window['accFrontal'].mean(),
            'std_frontal': window['accFrontal'].std(),
            'mean_vertical': window['accVertical'].mean(),
            'std_vertical': window['accVertical'].std(),
            'mean_lateral': window['accLateral'].mean(),
            'std_lateral': window['accLateral'].std(),
        }
        caracteristicas.append(feat)
        
    return pd.DataFrame(caracteristicas)

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file = request.files['file']
        filename = file.filename
        
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        base_name = os.path.splitext(clean_filename)[0]
        
        # Separar por espacios, guiones bajos o guiones medios de forma inteligente
        parts = re.split(r'[_\s-]+', base_name)
        activity_keywords = ['sentada', 'bipeda', 'marcha', 'acostada', 'caminar', 'correr', 'trotar']
        
        activity_name = "general"
        patient_parts = []
        
        for part in parts:
            part_lower = part.strip().lower()
            if part_lower in activity_keywords:
                activity_name = part_lower
            elif part_lower:
                patient_parts.append(part.strip())
                
        patient_id = " ".join(patient_parts).strip().title()
        if not patient_id or "Dataset" in patient_id:
            patient_id = "Mejia Karol"

        activity_mapping = {
            'acostada': '1',
            'sentada': '2',
            'bipeda': '3',
            'marcha': '4',
            'caminar': '4'
        }
        assigned_code = activity_mapping.get(activity_name, '1')

        # LECTURA BLINDADA
        file.seek(0)
        file_bytes = file.read()
        text_data = ""
        for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text_data = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        
        if not text_data:
            return jsonify({'error': 'No se pudo decodificar el archivo.'}), 400

        lines = text_data.splitlines()
        data_rows = []
        for line in lines:
            clean_line = line.replace(',', ' ').replace(';', ' ').replace('\t', ' ')
            tokens = clean_line.split()
            
            numeric_row = []
            for token in tokens:
                token_clean = ''.join([c for c in token if c.isdigit() or c in ['.', '-']])
                if token_clean and token_clean not in ['.', '-']:
                    try:
                        numeric_row.append(float(token_clean))
                    except ValueError:
                        continue
            
            if len(numeric_row) >= 4:
                data_rows.append(numeric_row[:4])

        if len(data_rows) == 0:
            return jsonify({'error': 'El archivo no contiene suficientes filas numéricas válidas.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral'])
        df['activityCode'] = assigned_code
        
        df = df.dropna().reset_index(drop=True)
        if len(df) == 0:
            return jsonify({'error': 'El archivo no contiene datos limpios.'}), 400

        activity_counts = df['activityCode'].value_counts().to_dict()
        total_records = len(df)
        
        start_time = df['time'].iloc[0] if 'time' in df.columns else 0
        end_time = df['time'].iloc[-1] if 'time' in df.columns else total_records / 10.0
        total_time = max(1, int(round(end_time - start_time)))

        gender = "Femenino" if any(n in patient_id.lower() for n in ['karol', 'maria', 'ana']) else "Masculino"

        pts_config = {'1': 0.5, '2': 0.0, '3': 2.0, '4': 2.0}
        score = 0
        active_records = 0
        
        for code, count in activity_counts.items():
            pts_per_sec = pts_config.get(str(code), 0.5)
            time_fraction = count / total_records
            score += (time_fraction * total_time) * pts_per_sec
            if str(code) in ['3', '4']:
                active_records += count

        score = int(round(score))
        active_percentage = float(round((active_records / total_records) * 100, 1)) if total_records > 0 else 0.0

        resultado = {
            "id": patient_id,
            "gender": gender,
            "totalRecords": total_records,
            "totalTime": total_time,
            "score": score,
            "activePercentage": active_percentage,
            "activityCounts": activity_counts,
            "data": df.to_dict(orient='records')
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
