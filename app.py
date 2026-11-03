import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "mensaje": "Backend activo y optimizado"})

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
        
        # Extraer género del último carácter del nombre (ej. d1p33F -> Femenino)
        gender_char = base_name[-1].upper() if len(base_name) > 0 else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        
        # Limpiar ID del paciente removiendo el carácter de género final
        patient_id_raw = base_name[:-1] if gender_char in ['F', 'M'] and len(base_name) > 1 and base_name[-2] in ['_', '-', ' '] else base_name
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
            if not line.strip():
                continue
            
            # Separar por comas, punto y coma o tabulaciones (CSV)
            if ',' in line:
                tokens = [t.strip() for t in line.split(',')]
            elif ';' in line:
                tokens = [t.strip() for t in line.split(';')]
            else:
                tokens = line.split()
            
            try:
                # Asegurar lectura estricta de las columnas requeridas:
                # Columna 1 (índice 0): Tiempo
                # Columna 2 (índice 1): Aceleración frontal
                # Columna 3 (índice 2): Aceleración vertical
                # Columna 4 (índice 3): Aceleración lateral
                # Columna 9 (índice 8): Etiqueta de actividad (1, 2, 3, 4)
                if len(tokens) >= 9:
                    t_val = float(tokens[0])
                    f_val = float(tokens[1])
                    v_val = float(tokens[2])
                    l_val = float(tokens[3])
                    act_val = int(float(tokens[8]))
                    data_rows.append([t_val, f_val, v_val, l_val, act_val])
            except ValueError:
                # Omite cabeceras de texto u otras filas no numéricas
                continue

        if len(data_rows) == 0:
            return jsonify({'error': 'El archivo no contiene filas numéricas válidas con el formato de 9 columnas.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral', 'activityCode'])
        df = df.dropna().reset_index(drop=True)
        total_records = len(df)
        
        # Normalización y cálculo preciso del tiempo en segundos
        t_min = df['time'].min()
        t_max = df['time'].max()
        raw_duration = t_max - t_min

        if raw_duration > 10000:
            # Si el tiempo viene en milisegundos
            total_time = int(round(raw_duration / 1000.0))
            df['time'] = ((df['time'] - t_min) / 1000.0).round(1)
        elif raw_duration > 0:
            # Si el tiempo viene en segundos directos
            total_time = int(round(raw_duration))
            df['time'] = (df['time'] - t_min).round(1)
        else:
            # Respaldo si los tiempos son estáticos o cero (asumiendo 50 Hz)
            total_time = max(1, int(round(total_records / 50.0)))
            df['time'] = [round(i / 50.0, 1) for i in range(total_records)]

        total_time = max(1, total_time)

        # Conteo exacto basado en la Columna 9 (activityCode)
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

        # Puntos por segundo según la actividad
        pts_config = {
            1: 0.5,
            2: 0.5,
            3: 0.2,
            4: 3.0
        }

        for code, count in sorted(activity_counts.items()):
            code_int = int(code)
            percentage = round((count / total_records) * 100, 1)
            est_time = max(1, int(round((count / total_records) * total_time)))
            
            pts_per_sec = pts_config.get(code_int, 1.0)
            points_earned = int(est_time * pts_per_sec * 10)
            total_score += points_earned
            
            if code_int == 4: # Deambulando se considera actividad activa
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
            "data": df.to_dict(orient='records')
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
