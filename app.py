import os
import re
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "mensaje": "Backend activo con formato exacto de 9 columnas"})

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if not request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        
        filename = file.filename if file.filename else "archivo.csv"
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        base_name = os.path.splitext(clean_filename)[0]
        
        # Extraer el género del último carácter del nombre del archivo (ej. d1p33F -> Femenino)
        gender_char = base_name[-1].upper() if len(base_name) > 0 else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        
        # Limpiar el ID del paciente removiendo el indicador de género final
        patient_id_raw = base_name[:-1] if gender_char in ['F', 'M'] and base_name[-2] in ['_', '-', ' '] else base_name
        patient_id = patient_id_raw.replace('_', ' ').replace('-', ' ').strip().upper()
        if not patient_id:
            patient_id = "PACIENTE 01"

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
            # Separar por comas, punto y coma o espacios (CSV)
            if ',' in line:
                tokens = [t.strip() for t in line.split(',')]
            elif ';' in line:
                tokens = [t.strip() for t in line.split(';')]
            else:
                tokens = line.split()
            
            numeric_row = []
            for token in tokens:
                token_clean = ''.join([c for c in token if c.isdigit() or c in ['.', '-']])
                if token_clean and token_clean not in ['.', '-']:
                    try:
                        numeric_row.append(float(token_clean))
                    except ValueError:
                        pass
            
            # Asegurar que leemos las columnas requeridas (al menos 9 columnas)
            if len(numeric_row) >= 9:
                t_val = numeric_row[0]  # Columna 1: Tiempo
                f_val = numeric_row[1]  # Columna 2: Frontal
                v_val = numeric_row[2]  # Columna 3: Vertical
                l_val = numeric_row[3]  # Columna 4: Lateral
                act_val = int(numeric_row[8]) # Columna 9: Etiqueta de actividad (índice 8)
                data_rows.append([t_val, f_val, v_val, l_val, act_val])
            elif len(numeric_row) >= 4:
                # Respaldo si el archivo viniera recortado
                data_rows.append([numeric_row[0], numeric_row[1], numeric_row[2], numeric_row[3], 1])

        if len(data_rows) == 0:
            return jsonify({'error': 'El archivo no contiene suficientes filas con el formato requerido.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral', 'activityCode'])
        df = df.dropna().reset_index(drop=True)
        total_records = len(df)
        
        start_time = df['time'].iloc[0] if 'time' in df.columns else 0
        end_time = df['time'].iloc[-1] if 'time' in df.columns else total_records / 10.0
        total_time = max(1, int(round(end_time - start_time)))

        # Conteo exacto basado en la Columna 9
        activity_counts = df['activityCode'].value_counts().to_dict()

        activity_names_map = {
            1: 'Sentado en la cama',
            2: 'Sentado en una silla',
            3: 'Acostado',
            4: 'Deambulando'
        }

        activity_breakdown = []
        total_score = 0
        active_records = 0

        # Puntos por segundo según la actividad (Deambular [4] otorga más puntos)
        pts_config = {
            1: 0.5,
            2: 0.5,
            3: 0.2,
            4: 3.0
        }

        for code, count in activity_counts.items():
            code_int = int(code)
            percentage = round((count / total_records) * 100, 1)
            est_time = max(1, int(round((count / total_records) * total_time)))
            
            pts_per_sec = pts_config.get(code_int, 1.0)
            points_earned = int(est_time * pts_per_sec * 10)
            total_score += points_earned
            
            if code_int == 4: # Actividad física (Deambulando)
                active_records += count

            act_label = activity_names_map.get(code_int, f"Actividad {code_int}")
            activity_breakdown.append({
                "activity": act_label,
                "time": f"{est_time} s",
                "percentage": f"{percentage}%",
                "points": f"+{points_earned} pts"
            })

        active_percentage = float(round((active_records / total_records) * 100, 1))

        badge = "Oro (Nivel Avanzado)" if total_score > 1500 else ("Plata (Nivel Intermedio)" if total_score > 800 else "Bronce (Nivel Iniciador)")

        resultado = {
            "id": patient_id,
            "gender": gender,
            "totalRecords": total_records,
            "totalTime": f"{total_time} s",
            "score": total_score,
            "badge": badge,
            "activePercentage": active_percentage,
            "activityBreakdown": activity_breakdown,
            "data": df.drop(columns=['activityCode']).to_dict(orient='records')
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
