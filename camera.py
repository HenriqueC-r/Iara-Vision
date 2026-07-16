import cv2
import os
import mediapipe as mp
import math


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Caminho do modelo
modelo = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)


# Configura o detector de mãos
base_options = python.BaseOptions(
    model_asset_path=modelo
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Abre a câmera DroidCam
camera = cv2.VideoCapture(0)

pinca_ativa = False


while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao acessar câmera")
        break


    # Converte BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Cria imagem para o MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # Detecta a mão
    resultado = detector.detect(mp_image)


    if resultado.hand_landmarks:

        for mao in resultado.hand_landmarks:

            polegar = mao[4]
            indicador = mao[8]

            distancia = math.hypot(
                polegar.x - indicador.x,
                polegar.y - indicador.y
            )

            if distancia < 0.05:
                if not pinca_ativa:
                    print("PINÇA")
                    pinca_ativa = True
            else:
                pinca_ativa = False

            # Desenha os pontos da mão
            for ponto in mao:

                x = int(ponto.x * frame.shape[1])
                y = int(ponto.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Desenha linhas entre os pontos
            conexoes = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for inicio, fim in conexoes:

                x1 = int(mao[inicio].x * frame.shape[1])
                y1 = int(mao[inicio].y * frame.shape[0])

                x2 = int(mao[fim].x * frame.shape[1])
                y2 = int(mao[fim].y * frame.shape[0])

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


        cv2.putText(
            frame,
            "MAO DETECTADA!",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


    cv2.imshow(
        "IARA Vision - Detector de Mao",
        frame
    )


    # Aperta Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



camera.release()
cv2.destroyAllWindows()