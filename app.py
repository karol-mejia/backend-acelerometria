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
else:
    modelo_rf = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "mensaje": "Backend activo"})

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if not request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        
        filename = file.filename if file.filename else "archivo.txt"
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        base_name = os.path.splitext(clean_filename)[0]
        
        # Limpieza inteligente: extraer nombre del paciente eliminando palabras clave de postura
        activity_keywords = ['sentada', 'bipeda', 'marcha', 'acostada', 'caminar', 'correr', 'trotar']
        parts = re.split(r'[_\s-]+', base_name)
        
        detected_activity = "General"
        patient_parts = []
        for part in parts:
            p_lower = part.strip().lower()
            if p_lower in activity_keywords:
                detected_activity = p_lower.capitalize()
            elif p_lower and not p_lower.isdigit():
                patient_parts.append(part.strip())
                
        # Unificar nombre base (ej. "Mejia Karol")
        patient_id = " ".join(patient_parts).strip().title()
        if not patient_id or len(patient_id) < 3:
            patient_id = "Mejia Karol"

        # Lectura robusta del archivo
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
            
            if len(numeric_row) >= 9:
                data_rows.append([numeric_row[2] if len(numeric_row) > 2 else len(data_rows)*0.1, numeric_row[3], numeric_row[4], numeric_row[5]])
            elif len(numeric_row) >= 4:
                data_rows.append(numeric_row[:4])
            elif len(numeric_row) == 3:
                data_rows.append([len(data_rows)*0.1, numeric_row[0], numeric_row[1], numeric_row[2]])

        if len(data_rows) == 0:
            return jsonify({'error': 'El archivo no contiene suficientes filas numéricas válidas.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral'])
        df = df.dropna().reset_index(drop=True)
        total_records = len(df)
        
        start_time = df['time'].iloc[0] if 'time' in df.columns else 0
        end_time = df['time'].iloc[-1] if 'time' in df.columns else total_records / 10.0
        total_time = max(1, int(round(end_time - start_time)))

        score = int(total_time * 35 if detected_activity in ['Bípeda', 'Marcha', 'Caminar'] else total_time * 15)
        active_percentage = 100.0 if detected_activity in ['Bípeda', 'Marcha', 'Caminar'] else 10.0

        activity_breakdown = [{
            "activity": f"Actividad: {detected_activity}",
            "time": f"{total_time} s",
            "percentage": f"{active_percentage}%",
            "points": f"+{score} pts"
        }]

        badge = "Oro (Nivel Avanzado)" if score > 1500 else ("Plata (Nivel Intermedio)" if score > 800 else "Bronce (Nivel Iniciador)")
        gender = "Femenino" if 'karol' in patient_id.lower() or 'f' in base_name.lower() else "Masculino"

        resultado = {
            "id": patient_id,
            "gender": gender,
            "totalRecords": total_records,
            "totalTime": f"{total_time} s",
            "score": score,
            "badge": badge,
            "activePercentage": active_percentage,
            "activityBreakdown": activity_breakdown,
            "data": df.to_dict(orient='records')
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
