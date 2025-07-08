import cv2
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

# Trayectorias
trayectoria_j = deque(maxlen=20)
trayectoria_z = deque(maxlen=20)

# === Funciones auxiliares ===
def dedo_estirado(landmarks, tip_id):
    return landmarks[tip_id].y < landmarks[tip_id - 2].y

def obtener_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0].landmark
    return None

def _no_detectado():
    return {"letra_detectada": "-", "score": 0.0, "es_correcto": False}

# === Detección letra J ===
def detectar_letra_j(frame, letra_seleccionada=None):
    if letra_seleccionada != 'J':
        return _no_detectado()

    landmarks = obtener_landmarks(frame)
    if landmarks:
        pinky = landmarks[20]
        trayectoria_j.append((pinky.x, pinky.y))
        trayectoria_z.clear()

        print(f"[J] Coordenadas pinky (x, y): {pinky.x:.4f}, {pinky.y:.4f}")
        print(f"[J] Trayectoria actual: {list(trayectoria_j)}")

        if not dedo_estirado(landmarks, 20): return _no_detectado()
        if dedo_estirado(landmarks, 8): return _no_detectado()
        if len(trayectoria_j) < 10: return _no_detectado()

        y = [p[1] for p in trayectoria_j]
        cambios = [y[i+1] - y[i] for i in range(len(y)-1)]
        bajadas = sum(1 for c in cambios if c > 0.01)
        subidas = sum(1 for c in cambios if c < -0.01)

        if bajadas >= 2 and subidas >= 1:
            print("[J] ✔ Letra J detectada con patrón de movimiento")
            trayectoria_j.clear()
            return {
                "letra_detectada": "J",
                "score": 1.0,
                "es_correcto": True
            }

    return _no_detectado()

# === Detección letra Z ===
def detectar_letra_z(frame, letra_seleccionada=None):
    if letra_seleccionada != 'Z':
        return _no_detectado()

    landmarks = obtener_landmarks(frame)
    if landmarks:
        index = landmarks[8]
        pinky = landmarks[20]
        trayectoria_z.append((index.x, index.y))
        trayectoria_j.clear()

        print(f"[Z] Coordenadas index (x, y): {index.x:.4f}, {index.y:.4f}")
        print(f"[Z] Trayectoria actual:", list(trayectoria_z))

        if not dedo_estirado(landmarks, 8): return _no_detectado()
        if dedo_estirado(landmarks, 20): return _no_detectado()
        if len(trayectoria_z) < 16: return _no_detectado()

        x = [p[0] for p in trayectoria_z]
        try:
            if x[0] < x[5] > x[10] < x[15] and abs(x[15] - x[0]) > 0.1:
                print("[Z] ✔ Letra Z detectada con patrón de movimiento")
                trayectoria_z.clear()
                return {
                    "letra_detectada": "Z",
                    "score": 1.0,
                    "es_correcto": True
                }
        except IndexError:
            pass

    return _no_detectado()
