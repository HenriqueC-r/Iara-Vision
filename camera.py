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

# Limiar de distância pra considerar "pinça fechada"
LIMIAR_PINCA = 0.05

# Tamanho do painel do menu (em pixels)
MENU_LARGURA = 260
MENU_ALTURA = 220

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao acessar câmera")
        break


    altura_frame, largura_frame = frame.shape[:2]

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

            # Menu fica aberto ENQUANTO a pinça estiver fechada
            # (sem toggle: solta a pinça e o menu some)
            pinca_fechada = distancia < LIMIAR_PINCA

            # Ponto de ancoragem do menu: o pulso (landmark 0)
            ancora_x = int(((polegar.x + indicador.x) / 2) * largura_frame) 
            ancora_y = int(((polegar.y + indicador.y) / 2) * altura_frame)

            # Desenha os pontos da mão
            for ponto in mao:

                x = int(ponto.x * largura_frame)
                y = int(ponto.y * altura_frame)

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

                x1 = int(mao[inicio].x * largura_frame)
                y1 = int(mao[inicio].y * altura_frame)

                x2 = int(mao[fim].x * largura_frame)
                y2 = int(mao[fim].y * altura_frame)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # Se a pinça estiver fechada, desenha o menu ancorado no pulso
            if pinca_fechada:

                # Posição do menu: acima e à direita do pulso
                x1_menu = ancora_x + 20
                y1_menu = ancora_y - MENU_ALTURA - 20
                x2_menu = x1_menu + MENU_LARGURA
                y2_menu = y1_menu + MENU_ALTURA

                # Garante que o menu não saia da tela (clamp)
                if x1_menu < 0:
                    x1_menu = 0
                    x2_menu = MENU_LARGURA
                if x2_menu > largura_frame:
                    x2_menu = largura_frame
                    x1_menu = largura_frame - MENU_LARGURA
                if y1_menu < 0:
                    y1_menu = 0
                    y2_menu = MENU_ALTURA
                if y2_menu > altura_frame:
                    y2_menu = altura_frame
                    y1_menu = altura_frame - MENU_ALTURA

                cv2.rectangle(
                    frame,
                    (x1_menu, y1_menu),
                    (x2_menu, y2_menu),
                    (40, 40, 40),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x1_menu, y1_menu),
                    (x2_menu, y2_menu),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "IARA Vision",
                    (x1_menu + 20, y1_menu + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Menu Principal",
                    (x1_menu + 20, y1_menu + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2
                )

                # Linha conectando o pulso ao menu (efeito "puxado da mao")
                cv2.line(
                    frame,
                    (ancora_x, ancora_y),
                    (x1_menu, y2_menu),
                    (0, 255, 255),
                    1
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