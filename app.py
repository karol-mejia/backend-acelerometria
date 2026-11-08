import os
import re
import io
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)
CORS(app)

def decodificar_texto(file_bytes):
    """Decodifica archivos soportando UTF-16, UTF-8 y Latin-1 (esencial para .txtux)."""
    encodings = ['utf-16', 'utf-16le', 'utf-16be', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            decoded = file_bytes.decode(enc)
            if decoded and len(decoded.strip()) > 0:
                if enc in ['utf-16', 'utf-16le', 'utf-16be', 'utf-8']:
                    return decoded, enc
                if decoded.count('\x00') < (len(decoded) * 0.05):
                    return decoded, enc
        except Exception:
            continue
    try:
        return file_bytes.decode('utf-16', errors='ignore'), 'utf-16 (forzado)'
    except:
        return file_bytes.decode('latin-1', errors='ignore'), 'latin-1 (forzado)'

def procesar_columnas_estrictas(file_bytes, filename):
    """
    Extrae estrictamente:
    - Columna 1 (Índice 0): Tiempo
    - Columna 2 (Índice 1): Aceleración Frontal
    - Columna 3 (Índice 2): Aceleración Vertical
    - Columna 4 (Índice 3): Aceleración Lateral
    - Columna 9 (Índice 8): Etiqueta / Código de Actividad
    """
    text_data, encoding_usado = decodificar_texto(file_bytes)
    lines = [line.strip() for line in text_data.splitlines() if line.strip()]
    
    if not lines:
        raise ValueError(f"El archivo {filename} está vacío.")

    # Detectar separador tabulado, coma o punto y coma
    sample = "\n".join(lines[:15])
    if sample.count('\t') > sample.count(','):
        sep = r'\t+'
    elif sample.count(';') > sample.count(','):
        sep = ';'
    elif sample.count(',') > 0:
        sep = ','
    else:
        sep = r'\s+'

    # Leer filas ignorando cabeceras de texto descriptivo
    data_rows = []
    for line in lines:
        l_low = line.lower()
        if any(w in l_low for w in ['time', 't(s)', 'acc', 'gyro', 'roll', 'pitch', 'yaw']):
            continue
        
        normalized = line.replace(',', '.')
        nums = re.findall(r'[-+]?\d*\.?\d+', normalized)
        
        # Necesitamos al menos las primeras 4 columnas (Tiempo y 3 ejes)
        if len(nums) >= 4:
            try:
                t_val = float(nums[0])  # Columna 1
                ax = float(nums[1])     # Columna 2
                ay = float(nums[2])     # Columna 3
                az = float(nums[3])     # Columna 4
                
                # Columna 9 (Índice 8). Si no existe en el archivo, se deduce del nombre.
                if len(nums) >= 9:
                    act_code = int(float(nums[8]))
                else:
                    b_lower = filename.lower()
                    if 'acostada' in b_lower: act_code = 3
                    elif 'sentada' in b_lower:
                        act_code = 1 if 'cama' in b_lower else 2
                    elif 'bipeda' in b_lower or 'bípeda' in b_lower: act_code = 2
                    elif 'marcha' in b_lower or 'caminar' in b_lower or 'deambulando' in b_lower: act_code = 4
                    else: act_code = 1
                
                data_rows.append([t_val, ax, ay, az, act_code])
            except Exception:
                continue

    if not data_rows:
        raise ValueError(f"El archivo {filename} no contiene filas numéricas válidas para las columnas requeridas.")

    df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral', 'activityCode'])
    df = df.dropna().reset_index(drop=True)
    return df, encoding_usado

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "mensaje": "Backend de Acelerometría con Random Forest activo"})

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

        file.seek(0)
        file_bytes = file.read()

        # Procesar datos reales usando columnas 1, 2, 3, 4 y 9
        df, encoding_usado = procesar_columnas_estrictas(file_bytes, filename)
        total_records = len(df)

        # Identificar paciente y género
        gender_char = base_name[-1].upper() if len(base_name) > 0 and base_name[-1].upper() in ['F', 'M'] else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        
        activity_keywords = ['sentada', 'bipeda', 'bípeda', 'marcha', 'acostada', 'caminar', 'correr', 'cama', 'silla', 'deambulando', 'datos']
        words = re.split(r'[_\s-]+', base_name)
        clean_words = [w.strip() for w in words if w.lower() not in activity_keywords and not w.isdigit() and len(w) > 0]
        patient_id = " ".join(clean_words).strip().title()
        if not patient_id or len(patient_id) < 2:
            match_p = re.search(r'(p\d+|d\d+p\d+)', base_name.lower())
            patient_id = f"Paciente {match_p.group(1).upper()}" if match_p else "Mejia Karol"

        # Entrenamiento y validación del modelo Random Forest con los datos reales
        rf_metrics = {}
        if total_records > 15:
            try:
                X = df[['accFrontal', 'accVertical', 'accLateral']]
                y = df['activityCode']
                
                # Evitar fuga de datos dividiendo train/test de forma estratificada si hay múltiples clases
                if len(y.unique()) > 1 and y.value_counts().min() >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                
                conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                rf_metrics = {
                    "status": "Entrenado con éxito",
                    "confusionMatrix": conf_matrix,
                    "featuresUsed": ["accFrontal", "accVertical", "accLateral"]
                }
            except Exception as rf_err:
                rf_metrics = {"status": "No se pudo entrenar (datos homogéneos)", "error": str(rf_err)}

        # Normalización temporal
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
        
        intensity_weights = {1: 0.3, 2: 0.5, 3: 0.1, 4: 1.0}
        pts_config = {1: 5, 2: 5, 3: 2, 4: 30}

        activity_breakdown = []
        total_score = 0
        weighted_activity_sum = 0

        for code, count in sorted(activity_counts.items()):
            code_int = int(code)
            percentage = round((count / total_records) * 100, 1)
            est_time = max(1, int(round((count / total_records) * total_time)))
            
            pts_per_sec = pts_config.get(code_int, 5)
            points_earned = int(est_time * (pts_per_sec / 10.0) * 10)
            total_score += points_earned
            
            weight = intensity_weights.get(code_int, 0.1)
            weighted_activity_sum += count * weight

            act_label = activity_names_map.get(code_int, f"Actividad {code_int}")
            activity_breakdown.append({
                "activity": act_label,
                "time": f"{est_time} s",
                "percentage": f"{percentage}%",
                "points": f"+{points_earned} pts",
                "code": code_int
            })

        active_percentage = float(round((weighted_activity_sum / total_records) * 100, 1))
        
        # Asignación de Insignias y Mensajes de Motivación basados en resultados reales
        if total_score > 1500:
            badge = "Oro (Nivel Avanzado)"
            motivation_message = "¡Excelente desempeño! Has mantenido un excelente equilibrio de movilidad y actividad física constante. ¡Sigue así!"
        elif total_score > 800:
            badge = "Plata (Nivel Intermedio)"
            motivation_message = "¡Muy buen trabajo! Tu constancia en las actividades cotidianas muestra un progreso sólido. ¡Intenta incorporar más caminatas!"
        else:
            badge = "Bronce (Nivel Iniciador)"
            motivation_message = "¡Buen inicio! Recuerda que cada movimiento cuenta. Incrementa poco a poco el tiempo en deambulación para ganar más puntos."

        resultado = {
            "id": patient_id,
            "gender": gender,
            "totalRecords": total_records,
            "totalTime": f"{total_time} s",
            "score": total_score,
            "badge": badge,
            "activePercentage": active_percentage,
            "motivationMessage": motivation_message,
            "randomForestMetrics": rf_metrics,
            "activityBreakdown": activity_breakdown,
            "data": df.to_dict(orient='records')
        }

        return jsonify(resultado)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 422

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
