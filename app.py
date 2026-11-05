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
    return jsonify({"status": "online", "mensaje": "Backend activo con parser Regex para .txtux"})

@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        if not request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        
        filename = file.filename if file.filename else "archivo.txt"
        clean_filename = filename.split('/')[-1].split('\\')[-1]
        
        base_name = clean_filename
        for ext in ['.txtux', '.txt', '.csv', '.dat', '.zip']:
            if base_name.lower().endswith(ext):
                base_name = base_name[:-len(ext)]
        
        # Extraer género del último carácter (ej. F o M)
        gender_char = base_name[-1].upper() if len(base_name) > 0 and base_name[-1].upper() in ['F', 'M'] else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        
        # Limpiar nombre del paciente agrupando todos los archivos bajo "Mejia Karol" o el nombre real
        activity_keywords = ['sentada', 'bipeda', 'marcha', 'acostada', 'caminar', 'correr', 'trotar', 'cama', 'silla', 'deambulando']
        patient_raw = base_name[:-1] if len(base_name) > 1 and base_name[-1].upper() in ['F', 'M'] else base_name
        
        clean_words = []
        for word in re.split(r'[_\s-]+', patient_raw):
            w_low = word.strip().lower()
            if w_low not in activity_keywords and not w_low.isdigit() and w_low not in ['f', 'm']:
                clean_words.append(word.strip())
                
        patient_id = " ".join(clean_words).strip().title()
        if not patient_id or len(patient_id) < 3:
            patient_id = "Mejia Karol"

        # Lectura robusta de bytes y decodificación
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
            line_str = line.strip()
            if not line_str:
                continue
            
            try:
                # Extraer TODOS los números válidos de la línea usando Regex (ignora símbolos y basura)
                numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line_str)
                
                if len(numbers) >= 9:
                    t_val = float(numbers[0])
                    f_val = float(numbers[1])
                    v_val = float(numbers[2])
                    l_val = float(numbers[3])
                    act_val = int(float(numbers[8])) # Columna 9 (índice 8)
                    data_rows.append([t_val, f_val, v_val, l_val, act_val])
                elif len(numbers) >= 4:
                    t_val = float(numbers[0])
                    f_val = float(numbers[1])
                    v_val = float(numbers[2])
                    l_val = float(numbers[3])
                    # Detección de actividad por el nombre del archivo si falta la columna 9
                    default_act = 1
                    b_lower = base_name.lower()
                    if 'bipeda' in b_lower: default_act = 3
                    elif 'marcha' in b_lower or 'caminar' in b_lower: default_act = 4
                    elif 'acostada' in b_lower: default_act = 3
                    elif 'sentada' in b_lower: default_act = 2
                    data_rows.append([t_val, f_val, v_val, l_val, default_act])
            except Exception:
                continue

        if len(data_rows) == 0:
            return jsonify({'error': f'El archivo {filename} no contiene filas numéricas procesables.'}), 400

        df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral', 'activityCode'])
        df = df.dropna().reset_index(drop=True)
        total_records = len(df)
        
        # Normalización de tiempo en segundos desde 0
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

        activity_counts = df['activityCode'].value_counts().to_dict()

        activity_names_map = {
            1: 'Sentado en la cama',
            2: 'Sentado en una silla',
            3: 'Acostado',
            4: 'Deambulando'
        }

        intensity_weights = {
            1: 0.3,
            2: 0.5,
            3: 0.1,
            4: 1.0
        }

        activity_breakdown = []
        total_score = 0
        weighted_activity_sum = 0

        pts_config = {
            1: 5,
            2: 5,
            3: 2,
            4: 30
        }

        for code, count in sorted(activity_counts.items()):
            code_int = int(code)
            percentage = round((count / total_records) * 100, 1)
            est_time = max(1, int(round((count / total_records) * total_time)))
            
            pts_per_sec = pts_config.get(code_int, 1.0)
            points_earned = int(est_time * (pts_per_sec / 10.0) * 10)
            total_score += points_earned
            
            weight = intensity_weights.get(code_int, 0.1)
            weighted_activity_sum += count * weight

            act_label = activity_names_map.get(code_int, f"Actividad {code_int}")
            activity_breakdown.append({
                "activity": act_label,
                "time": f"{est_time} s",
                "percentage": f"{percentage}%",
                "points": f"+{points_earned} pts"
            })

        active_percentage = float(round((weighted_activity_sum / total_records) * 100, 1))
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
