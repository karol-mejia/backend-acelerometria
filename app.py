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
    return jsonify({"status": "online", "mensaje": "Backend activo con soporte UTF-16 para .txtux"})

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if not request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        
        filename = file.filename if file.filename else "archivo.txt"
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        b_lower = clean_filename.lower()
        
        # Determinar actividad por el nombre del archivo
        if 'acostada' in b_lower:
            activity_code = 3
            act_name = 'Acostado'
        elif 'sentada' in b_lower:
            activity_code = 2
            act_name = 'Sentado en una silla'
        elif 'bipeda' in b_lower or 'bípeda' in b_lower:
            activity_code = 3
            act_name = 'Bípeda (Estática)'
        elif 'marcha' in b_lower or 'caminar' in b_lower:
            activity_code = 4
            act_name = 'Deambulando'
        else:
            activity_code = 1
            act_name = 'Sentado en la cama'

        gender_char = clean_filename[-1].upper() if len(clean_filename) > 0 and clean_filename[-1].upper() in ['F', 'M'] else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        patient_id = "Mejia Karol"

        # Lectura robusta con soporte prioritario para UTF-16 (archivos .txtux)
        file.seek(0)
        file_bytes = file.read()
        text_data = ""
        for enc in ['utf-16', 'utf-16le', 'utf-16be', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text_data = file_bytes.decode(enc)
                if len(text_data.strip()) > 0 and '\x00' not in text_data[:100]:
                    break
            except UnicodeDecodeError:
                continue
        
        if not text_data:
            return jsonify({'error': f'No se pudo decodificar el archivo {filename}.'}), 400

        lines = text_data.splitlines()
        data_rows = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            
            line_lower = line_str.lower()
            if any(word in line_lower for word in ['time', 't(s)', 'acc', 'gyro', 'roll', 'pitch', 'yaw']):
                continue
            
            try:
                # Normalizar comas decimales a puntos y extraer números
                normalized_line = line_str.replace(',', '.')
                numbers = re.findall(r'[-+]?\d*\.?\d+', normalized_line)
                
                # Se esperan columnas: Time, AccX, AccY, AccZ (al menos 4 números)
                if len(numbers) >= 4:
                    t_val = float(numbers[0])
                    ax = float(numbers[1])
                    ay = float(numbers[2])
                    az = float(numbers[3])
                    
                    data_rows.append([t_val, ax, ay, az, activity_code])
            except Exception:
                continue

        if len(data_rows) == 0:
            return jsonify({'error': f'El archivo {filename} no contiene filas numéricas válidas.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral', 'activityCode'])
        df = df.dropna().reset_index(drop=True)
        total_records = len(df)
        
        t_min = df['time'].min()
        t_max = df['time'].max()
        raw_duration = t_max - t_min

        if raw_duration > 10000:
            total_time = int(round(raw_duration / 1000.0))
            df['time'] = ((df['time'] - t_min) / 1000.0).round(1)
        elif raw_duration > 0:
            total_time = int(round(raw_duration))
            df['time'] = (df['time'] - t_min).round(1)
        else:
            total_time = max(1, int(round(total_records / 50.0)))
            df['time'] = [round(i / 50.0, 1) for i in range(total_records)]

        total_time = max(1, total_time)

        est_time = total_time
        percentage = 100.0
        
        pts_config = {1: 5, 2: 5, 3: 2, 4: 30}
        pts_per_sec = pts_config.get(activity_code, 5)
        points_earned = int(est_time * (pts_per_sec / 10.0) * 10)
        total_score = points_earned

        activity_breakdown = [{
            "activity": act_name,
            "time": f"{est_time} s",
            "percentage": f"{percentage}%",
            "points": f"+{points_earned} pts"
        }]

        intensity_weights = {1: 0.3, 2: 0.5, 3: 0.1, 4: 1.0}
        active_percentage = float(intensity_weights.get(activity_code, 0.5) * 100)
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
