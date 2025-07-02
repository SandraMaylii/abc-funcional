from flask import Flask, render_template, request, jsonify, Response
import base64
import cv2
import os
import numpy as np
import requests
from detect import detectar_letra  
from mediapipe_gestures import detectar_letra_j, detectar_letra_z  
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

app = Flask(__name__)

# === Variables globales ===
buffer_letras = ""
palabras_sugeridas = []
palabra_elegida = ""
ultima_letra = None

# === Sugerencias automáticas ===
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import requests

# Modelo para letras distintas de "w"
model_id = "PlanTL-GOB-ES/roberta-base-bne"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=20)

# Modelo especial solo para "w"
gpt_generator = pipeline("text-generation", model="PlanTL-GOB-ES/gpt2-base-bne")

def sugerir_palabra(letras, topn=5):
    letras = letras.strip().lower()
    if not letras:
        return []

    if letras.startswith("w"):
        try:
            results = gpt_generator(letras, max_length=10, num_return_sequences=topn)
            sugerencias = [
                res["generated_text"].split()[0].strip()
                for res in results
                if res["generated_text"].lower().startswith(letras)
            ]
            return list(dict.fromkeys(sugerencias))[:topn]  # Elimina duplicados
        except Exception as e:
            print("Error en sugerencia GPT Hugging Face:", e)
            return []

    # Resto de letras: usa fill-mask y Datamuse
    input_text = f"{letras}<mask>"
    try:
        results = fill_mask(input_text)
        sugerencias = [
            r['token_str'].replace(' ', '').strip()
            for r in results
            if r['token_str'].lower().startswith(letras)
        ]
        if sugerencias:
            return list(dict.fromkeys(sugerencias))[:topn]
    except Exception as e:
        print("Error en sugerencia :", e)

    # Fallback: Datamuse
    url = f"https://api.datamuse.com/words?sp={letras}*&v=es&max={topn}"
    try:
        response = requests.get(url)
        return [item['word'] for item in response.json()] if response.status_code == 200 else []
    except:
        return []


# === Agregar letra al buffer si es válida ===
def agregar_letra(letra_detectada, threshold=0.75, probabilidad=1.0):
    global buffer_letras, ultima_letra, palabras_sugeridas
    if probabilidad >= threshold and (letra_detectada != ultima_letra or ultima_letra is None):
        ultima_letra = letra_detectada
        buffer_letras += letra_detectada
        palabras_sugeridas = sugerir_palabra(buffer_letras, topn=8)
        print(f"[✅] Letra agregada al buffer: {letra_detectada}")
    else:
        print(f"[❌] No se agregó letra. Prob: {probabilidad}, Última: {ultima_letra}")

# === Página principal ===
@app.route('/')
def index():
    return render_template('index.html')

# === Endpoint para letras A–Y (excepto J y Z) ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    letra = data['letra']  

    # Si es J o Z, redirigir al endpoint correcto
    if letra in ['J', 'Z']:
        return jsonify({"letra_detectada": "-", "score": 0.0, "es_correcto": False})

    # Procesar imagen
    encoded = img_data.split(',')[1]
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resultado = detectar_letra(frame, letra)
    letra_detectada = resultado["letra_detectada"]
    score = resultado["score"]

    print(f"[INFO] Letra seleccionada: {letra}")
    print(f"[INFO] Letra detectada: {letra_detectada} con score {score}")

    if letra_detectada != "-" and score >= 0.75:
        agregar_letra(letra_detectada, probabilidad=score)
    else:
        print("[❌] No se agregó la letra por baja precisión o error.")

    return jsonify(resultado)

# === Endpoint especial para J y Z ===
@app.route('/predict_jz', methods=['POST'])
def predict_jz():
    data = request.json
    img_data = data['image']
    letra = data['letra']

    encoded = img_data.split(',')[1]
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if letra == 'J':
        resultado = detectar_letra_j(frame, letra)
    elif letra == 'Z':
        resultado = detectar_letra_z(frame, letra)
    else:
        resultado = {"letra_detectada": "-", "score": 0.0, "es_correcto": False}

    letra_detectada = resultado.get("letra_detectada", "-")
    score = resultado.get("score", 0.0)
    es_correcto = resultado.get("es_correcto", False)

    print(f"[JZ] Letra detectada: {letra_detectada} | Score: {score} | Correcto: {es_correcto}")

    if letra_detectada != "-" and score >= 0.75 and es_correcto:
        agregar_letra(letra_detectada, probabilidad=score)

    return jsonify(resultado)

# === Endpoints auxiliares ===
@app.route('/texto_actual')
def texto_actual():
    return jsonify(buffer_letras)

@app.route('/sugerencias')
def get_sugerencias():
    return jsonify(palabras_sugeridas)

@app.route('/seleccionar')
def seleccionar():
    global palabra_elegida, buffer_letras, ultima_letra
    palabra = request.args.get("palabra", "")
    if palabra:
        palabra_elegida += palabra + " "
    buffer_letras = ""
    ultima_letra = None
    return Response("ok")

@app.route('/palabra_seleccionada')
def palabra_seleccionada():
    return jsonify(palabra_elegida)

@app.route('/borrar_ultima')
def borrar_ultima():
    global buffer_letras, palabras_sugeridas, ultima_letra
    if buffer_letras:
        buffer_letras = buffer_letras[:-1]
        palabras_sugeridas = sugerir_palabra(buffer_letras, topn=8) if buffer_letras else []
        ultima_letra = None
    return Response("ok")

@app.route('/borrar_texto_completo')
def borrar_texto_completo():
    global buffer_letras, palabras_sugeridas
    buffer_letras = ""
    palabras_sugeridas = []
    return Response("ok")

@app.route('/borrar_ultima_seleccionada')
def borrar_ultima_seleccionada():
    global palabra_elegida
    palabras = palabra_elegida.strip().split()
    if palabras:
        palabras.pop()
    palabra_elegida = ' '.join(palabras) + (" " if palabras else "")
    return Response("ok")

@app.route('/borrar_seleccionadas')
def borrar_seleccionadas():
    global palabra_elegida
    palabra_elegida = ""
    return Response("ok")

@app.route('/limpiar')
def limpiar():
    global buffer_letras, palabras_sugeridas, palabra_elegida, ultima_letra, texto_escrito
    buffer_letras = ""
    palabras_sugeridas.clear()
    palabra_elegida = ""
    ultima_letra = None
    texto_escrito = "" 
    return Response("ok")

@app.route('/estado')
def estado():
    return jsonify({
        "texto": buffer_letras,
        "sugerencias": palabras_sugeridas,
        "seleccionada": palabra_elegida
    })



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Puerto que asigna Cloud Run
    app.run(host='0.0.0.0', port=port, debug=False)

