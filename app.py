import os
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
        
        # Extraer nombre del paciente si viene en formato personalizado
        parts = base_name.replace('-', '_').split('_')
        activity_keywords = ['sentada', 'bipeda', 'marcha', 'acostada', 'caminar', 'correr', 'trotar']
        
        activity_name = "general"
        patient_parts = []
        
        for part in parts:
            part_lower = part.strip().lower()
            if part_lower in activity_keywords:
                activity_name = part_lower
            else:
                patient_parts.append(part.strip())
                
        patient_id = " ".join(patient_parts).strip().title()
        if not patient_id or "Dataset" in patient_id:
            patient_id = "Paciente Dataset"

        activity_mapping = {
            'acostada': '1',
            'sentada': '2',
            'bipeda': '3',
            'marcha': '4',
            'caminar': '4'
        }
        assigned_code = activity_mapping.get(activity_name, '1')

        # LECTURA INTELIGENTE DEL ARCHIVO CSV / TXT
        df = None
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', comment='#', on_bad_lines='skip', header=None)
        except Exception:
            pass

        if df is not None and not df.empty:
            if df.shape[1] >= 9:
                # Estructura del Dataset oficial (9 columnas):
                # Col 1 (idx 0): Time, Col 2-4 (idx 1-3): Aceleraciones, Col 9 (idx 8): Label de actividad
                new_df = pd.DataFrame()
                new_df['time'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                new_df['accFrontal'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                new_df['accVertical'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                new_df['accLateral'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
                new_df['activityCode'] = df.iloc[:, 8].astype(str).str.strip()
                df = new_df.dropna(subset=['time', 'accFrontal', 'accVertical', 'accLateral']).reset_index(drop=True)
            elif df.shape[1] >= 4:
                # Estructura personalizada (4 columnas o más sin etiqueta en la col 9)
                new_df = pd.DataFrame()
                new_df['time'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                new_df['accFrontal'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                new_df['accVertical'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                new_df['accLateral'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
                new_df['activityCode'] = assigned_code
                df = new_df.dropna().reset_index(drop=True)

        # Fallback por si acaso el archivo viene con separadores mixtos o espacios
        if df is None or df.empty or len(df) == 0:
            file.seek(0)
            file_content = file.read()
            text_data = file_content.decode('utf-8', errors='ignore') if isinstance(file_content, bytes) else str(file_content)
            
            lines = text_data.splitlines()
            data_rows = []
            for line in lines:
                clean_line = line.replace(',', ' ').replace(';', ' ').replace('\t', ' ')
                tokens = clean_line.split()
                numeric_row = [float(t) for t in tokens if t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())]
                if len(numeric_row) >= 4:
                    data_rows.append(numeric_row[:4])
            
            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral'])
                df['activityCode'] = assigned_code

        if df is None or df.empty or len(df) == 0:
            return jsonify({'error': 'El archivo no contiene suficientes columnas numéricas válidas.'}), 400

        # Si el modelo Random Forest está disponible y no tenemos etiquetas de columna 9, aplicamos predicción
        if modelo_rf is not None and (df['activityCode'].nunique() <= 1) and len(df) >= 100:
            X_nuevo = extraer_caracteristicas_ventana(df)
            if not X_nuevo.empty:
                predicciones = modelo_rf.predict(X_nuevo)
                activity_codes = []
                for pred in predicciones:
                    activity_codes.extend([str(pred)] * 100)
                if len(activity_codes) < len(df):
                    activity_codes.extend([str(predicciones[-1])] * (len(df) - len(activity_codes)))
                else:
                    activity_codes = activity_codes[:len(df)]
                df['activityCode'] = activity_codes

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
