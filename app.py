import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite la comunicación con tu página web

# CARGAR EL MODELO RANDOM FOREST AL INICIAR EL SERVIDOR
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
        "mensaje": "¡El backend de acelerometría y Random Forest está funcionando correctamente!"
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
            return jsonify({'error': 'No se encontró ningún archivo en la petición'}), 400
        
        file = request.files['file']
        filename = file.filename
        
        # Extraer nombre del paciente y actividad del nombre del archivo (ej: Mejia Karol_sentada.txt)
        base_name = os.path.splitext(filename)[0]
        parts = base_name.replace('-', '_').split('_')
        
        if len(parts) >= 2:
            activity_name = parts[-1].strip().lower()
            patient_id = " ".join(parts[:-1]).strip().title()
        else:
            patient_id = base_name.strip().title()
            activity_name = "general"

        # Mapear nombre de actividad a código estándar
        activity_mapping = {
            'acostada': '1',
            'sentada': '2',
            'bipeda': '3',
            'marcha': '4',
            'caminar': '4'
        }
        assigned_code = activity_mapping.get(activity_name, '1')

        # LECTURA ULTRA-ROBUSTA: Soporta cualquier separador (comas, tabs, espacios, etc.)
        df = None
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', comment='#', on_bad_lines='skip')
        except Exception:
            pass

        if df is None or df.empty or df.shape[1] < 4:
            file.seek(0)
            try:
                file_content = file.read()
                if isinstance(file_content, bytes):
                    text_data = file_content.decode('utf-8', errors='ignore')
                else:
                    text_data = str(file_content)
                
                lines = text_data.splitlines()
                data_rows = []
                for line in lines:
                    clean_line = line.replace(',', ' ').replace(';', ' ').replace('\t', ' ')
                    tokens = clean_line.split()
                    
                    numeric_row = []
                    for token in tokens:
                        try:
                            numeric_row.append(float(token))
                        except ValueError:
                            continue
                    
                    if len(numeric_row) >= 4:
                        data_rows.append(numeric_row[:4])
                
                if len(data_rows) > 0:
                    df = pd.DataFrame(data_rows, columns=['time', 'accFrontal', 'accVertical', 'accLateral'])
            except Exception as parse_err:
                return jsonify({'error': f'No se pudo interpretar el formato del archivo: {str(parse_err)}'}), 400

        if df is None or df.empty or df.shape[1] < 4:
            return jsonify({'error': 'El archivo no contiene suficientes columnas numéricas válidas.'}), 400

        df = df.iloc[:, :4]
        df.columns = ['time', 'accFrontal', 'accVertical', 'accLateral']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)

        if len(df) == 0:
            return jsonify({'error': 'El archivo no contiene datos numéricos limpios.'}), 400

        # Aplicar modelo Random Forest o asignar código por nombre de archivo
        if modelo_rf is not None and len(df) >= 100:
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
            else:
                df['activityCode'] = assigned_code
        else:
            df['activityCode'] = assigned_code

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
            pts_per_sec = pts_config.get(code, 0.5)
            time_fraction = count / total_records
            score += (time_fraction * total_time) * pts_per_sec
            if code == '3' or code == '4':
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
        print(f"Error procesando el archivo en el servidor: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

