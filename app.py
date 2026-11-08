import os
import re
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def decodificar_texto(file_bytes):
    """Prueba múltiples codificaciones para decodificar bytes sin corromper el contenido."""
    encodings = ['utf-16', 'utf-16le', 'utf-16be', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            text = file_bytes.decode(enc)
            # Verificar si no contiene demasiados caracteres nulos de decodificación fallida
            if len(text.strip()) > 0 and text.count('\x00') < (len(text) * 0.1):
                return text, enc
        except (UnicodeDecodeError, Exception):
            continue
    return None, None

def mapear_columnas_inteligente(df_raw, filename):
    """
    Analiza un DataFrame sin estructura fija e identifica automáticamente:
    - Columna de Tiempo
    - Ejes de Aceleración (Frontal/X, Vertical/Y, Lateral/Z)
    - Columna o Estado de Actividad
    """
    columnas = [str(col).strip() for col in df_raw.columns]
    num_cols = len(columnas)
    
    col_map = {
        'time': None,
        'accFrontal': None,
        'accVertical': None,
        'accLateral': None,
        'activity': None
    }

    # Patterns Regex para detectar nombres habituales
    patterns_time = r'^(time|tiempo|t|t\(s\)|timestamp|sec|segundos)$'
    patterns_x = r'^(acc_?x|x| frontal|accfrontal|x\(m/s\^2\)|ax)$'
    patterns_y = r'^(acc_?y|y|vertical|accvertical|y\(m/s\^2\)|ay)$'
    patterns_z = r'^(acc_?z|z|lateral|acclateral|z\(m/s\^2\)|az)$'
    patterns_act = r'^(activity|actividad|label|code|codigo|estado|postura)$'

    # 1. Búsqueda por encabezados
    for idx, col in enumerate(columnas):
        c_low = col.lower().strip()
        if not col_map['time'] and re.search(patterns_time, c_low):
            col_map['time'] = idx
        elif not col_map['accFrontal'] and re.search(patterns_x, c_low):
            col_map['accFrontal'] = idx
        elif not col_map['accVertical'] and re.search(patterns_y, c_low):
            col_map['accVertical'] = idx
        elif not col_map['accLateral'] and re.search(patterns_z, c_low):
            col_map['accLateral'] = idx
        elif not col_map['activity'] and re.search(patterns_act, c_low):
            col_map['activity'] = idx

    # 2. Asignación por posición/orden si no se detectó por nombre
    numeric_cols = []
    for idx in range(num_cols):
        # Convertir a numérico para evaluar el tipo de variable
        series = pd.to_numeric(df_raw.iloc[:, idx].astype(str).str.replace(',', '.'), errors='coerce')
        if series.notnull().sum() > (len(df_raw) * 0.5):
            numeric_cols.append(idx)

    if col_map['time'] is None and len(numeric_cols) >= 1:
        col_map['time'] = numeric_cols[0]

    # Asignar ejes secuencialmente a las siguientes columnas numéricas
    remaining_num = [c for c in numeric_cols if c != col_map['time']]
    
    if col_map['accFrontal'] is None and len(remaining_num) >= 1:
        col_map['accFrontal'] = remaining_num[0]
    if col_map['accVertical'] is None and len(remaining_num) >= 2:
        col_map['accVertical'] = remaining_num[1]
    if col_map['accLateral'] is None and len(remaining_num) >= 3:
        col_map['accLateral'] = remaining_num[2]

    # Detectar actividad si hay una 9na columna o por descarte de columna entera/categorica
    if col_map['activity'] is None:
        if num_cols >= 9:
            col_map['activity'] = 8 # Columna 9 (índice 8)
        else:
            for c in remaining_num[3:]:
                series = pd.to_numeric(df_raw.iloc[:, c].astype(str).str.replace(',', '.'), errors='coerce')
                unique_vals = series.dropna().unique()
                if len(unique_vals) <= 10 and all(v.is_integer() for v in unique_vals if pd.notnull(v)):
                    col_map['activity'] = c
                    break

    return col_map, columnas

def parse_sensor_dataframe(text_data, filename):
    """Parsea el texto en un DataFrame estructurado identificando separadores y encabezados."""
    lines = [line.strip() for line in text_data.splitlines() if line.strip()]
    if not lines:
        raise ValueError("El archivo está completamente vacío.")

    # Detectar el separador probabilísticamente
    sample_block = "\n".join(lines[:20])
    comma_count = sample_block.count(',')
    semicolon_count = sample_block.count(';')
    tab_count = sample_block.count('\t')

    if tab_count > comma_count and tab_count > semicolon_count:
        sep = r'\t+'
    elif semicolon_count > comma_count:
        sep = ';'
    elif comma_count > 0:
        sep = ','
    else:
        sep = r'\s+'

    # Intentar leer con pandas
    try:
        df_raw = pd.read_csv(io.StringIO(text_data), sep=sep, engine='python', on_bad_lines='skip')
    except Exception:
        # Respaldo con regex fila por fila
        data_matrix = []
        for l in lines:
            normalized = l.replace(',', '.')
            nums = re.findall(r'[-+]?\d*\.?\d+', normalized)
            if nums:
                data_matrix.append(nums)
        df_raw = pd.DataFrame(data_matrix)

    if df_raw.empty:
        raise ValueError("No se pudieron extraer datos o filas estructuradas del archivo.")

    # Mapear columnas
    col_map, col_names = mapear_columnas_inteligente(df_raw, filename)

    # Validaciones fundamentales
    missing_elements = []
    if col_map['time'] is None:
        missing_elements.append("Columna de Tiempo (sec/timestamp)")
    if col_map['accFrontal'] is None:
        missing_elements.append("Eje de Aceleración Principal (Frontal/X)")

    if missing_elements:
        raise ValueError(f"Estructura no reconocida o incompleta. No se detectó: {', '.join(missing_elements)}.")

    # Extraer y limpiar DataFrame normalizado
    df_clean = pd.DataFrame()
    df_clean['time'] = pd.to_numeric(df_raw.iloc[:, col_map['time']].astype(str).str.replace(',', '.'), errors='coerce')
    df_clean['accFrontal'] = pd.to_numeric(df_raw.iloc[:, col_map['accFrontal']].astype(str).str.replace(',', '.'), errors='coerce')
    
    if col_map['accVertical'] is not None:
        df_clean['accVertical'] = pd.to_numeric(df_raw.iloc[:, col_map['accVertical']].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df_clean['accVertical'] = df_clean['accFrontal']

    if col_map['accLateral'] is not None:
        df_clean['accLateral'] = pd.to_numeric(df_raw.iloc[:, col_map['accLateral']].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df_clean['accLateral'] = df_clean['accFrontal']

    # Asignación de código de actividad (por columna o por nombre del archivo)
    b_lower = filename.lower()
    default_act = 1
    if 'acostada' in b_lower: default_act = 3
    elif 'sentada' in b_lower: default_act = 2
    elif 'bipeda' in b_lower or 'bípeda' in b_lower: default_act = 2
    elif 'marcha' in b_lower or 'caminar' in b_lower or 'deambulando' in b_lower: default_act = 4

    if col_map['activity'] is not None:
        act_series = pd.to_numeric(df_raw.iloc[:, col_map['activity']].astype(str).str.replace(',', '.'), errors='coerce')
        df_clean['activityCode'] = act_series.fillna(default_act).astype(int)
    else:
        df_clean['activityCode'] = default_act

    df_clean = df_clean.dropna(subset=['time', 'accFrontal']).reset_index(drop=True)

    if df_clean.empty:
        raise ValueError("El archivo contiene encabezados pero ninguna fila con datos numéricos válidos.")

    meta_info = {
        'total_filas_originales': len(df_raw),
        'total_filas_validas': len(df_clean),
        'columnas_detectadas': col_names,
        'mapeo_indices': col_map,
        'separador_utilizado': str(sep)
    }

    return df_clean, meta_info

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online", "mensaje": "Backend de Ingesta Inteligente Activo"})

@app.route('/inspeccionar', methods=['POST'])
def inspeccionar():
    """Paso 2, 3 y 6: Inspecciona un archivo sin procesarlo definitivamente y entrega vista previa."""
    try:
        if not request.files:
            return jsonify({'error': 'No se recibió ningún archivo'}), 400

        file_key = list(request.files.keys())[0]
        file = request.files[file_key]
        filename = file.filename if file.filename else "archivo.txt"

        file.seek(0)
        file_bytes = file.read()
        text_data, encoding_used = decodificar_texto(file_bytes)

        if not text_data:
            return jsonify({'error': f'Formato no compatible. No se pudo decodificar la codificación del archivo {filename}.'}), 400

        df_clean, meta = parse_sensor_dataframe(text_data, filename)

        # Generar vista previa de las primeras 5 filas
        preview_rows = df_clean.head(5).to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'filename': filename,
            'encoding': encoding_used,
            'meta': meta,
            'preview': preview_rows
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 422

@app.route('/analizar', methods=['POST'])
def analizar():
    """Pasos 4 y 5: Procesa los datos dinámicamente y entrega las métricas globales."""
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
        text_data, _ = decodificar_texto(file_bytes)

        if not text_data:
            return jsonify({'error': f'No se pudo decodificar el archivo {filename}.'}), 400

        df, meta = parse_sensor_dataframe(text_data, filename)
        total_records = len(df)

        # Identificación del paciente
        gender_char = base_name[-1].upper() if len(base_name) > 0 and base_name[-1].upper() in ['F', 'M'] else 'F'
        gender = "Femenino" if gender_char == 'F' else "Masculino"
        
        activity_keywords = ['sentada', 'bipeda', 'bípeda', 'marcha', 'acostada', 'caminar', 'correr', 'cama', 'silla', 'deambulando', 'datos']
        words = re.split(r'[_\s-]+', base_name)
        clean_words = [w.strip() for w in words if w.lower() not in activity_keywords and not w.isdigit() and len(w) > 0]
        patient_id = " ".join(clean_words).strip().title()
        if not patient_id or len(patient_id) < 2:
            patient_id = "Mejia Karol"

        # Normalización del tiempo
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
            "data": df.to_dict(orient='records'),
            "meta": meta
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 422

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
