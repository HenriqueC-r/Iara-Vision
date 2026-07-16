import cv2
import os
import mediapipe as mp
import math
import time


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

# Tempo de espera (segundos) segurando a pinça pra abrir o menu
TEMPO_ABERTURA = 3.0

# Tamanho do painel do menu (em pixels)
MENU_LARGURA = 260
MENU_ALTURA = 220

# Raio do anel de progresso
RAIO_ANEL = 40


# ---- Estado do menu (persiste entre frames) ----
menu_aberto = False
menu_pos = None            # (x1, y1, x2, y2) travado quando o menu abre
tempo_inicio_pinca = None  # timestamp de quando a pinça fechou
pinca_fechada_anterior = False


def desenhar_anel_progresso(frame, cx, cy, progresso):
    """Desenha o anel de carregamento (0.0 a 1.0) ao redor do ponto da pinça."""

    # Anel de fundo (cinza)
    cv2.circle(
        frame,
        (cx, cy),
        RAIO_ANEL,
        (80, 80, 80),
        4
    )

    # Arco de progresso (amarelo), começando do topo (-90 graus)
    angulo_final = int(360 * progresso)

    cv2.ellipse(
        frame,
        (cx, cy),
        (RAIO_ANEL, RAIO_ANEL),
        -90,
        0,
        angulo_final,
        (0, 255, 255),
        6
    )


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

    pinca_fechada = False
    ancora_x, ancora_y = None, None

    if resultado.hand_landmarks:

        for mao in resultado.hand_landmarks:

            polegar = mao[4]
            indicador = mao[8]

            distancia = math.hypot(
                polegar.x - indicador.x,
                polegar.y - indicador.y
            )

            pinca_fechada = distancia < LIMIAR_PINCA

            # Ponto de ancoragem: o meio da pinça (entre polegar e indicador)
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

            # Só processa a lógica de menu pra primeira mão detectada
            break

        cv2.putText(
            frame,
            "MAO DETECTADA!",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


    # ---- Máquina de estados do menu ----

    if not menu_aberto:

        if pinca_fechada:

            if tempo_inicio_pinca is None:
                tempo_inicio_pinca = time.time()

            decorrido = time.time() - tempo_inicio_pinca
            progresso = min(decorrido / TEMPO_ABERTURA, 1.0)

            # Desenha o anel de carregamento na posição atual da pinça
            if ancora_x is not None:
                desenhar_anel_progresso(frame, ancora_x, ancora_y, progresso)

            if decorrido >= TEMPO_ABERTURA:

                # Abre o menu e TRAVA a posição (fica fixo, não segue mais a mão)
                x1_menu = ancora_x + 20
                y1_menu = ancora_y - MENU_ALTURA - 20
                x2_menu = x1_menu + MENU_LARGURA
                y2_menu = y1_menu + MENU_ALTURA

                # Clamp pra não sair da tela
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

                menu_pos = (x1_menu, y1_menu, x2_menu, y2_menu)
                menu_aberto = True

        else:
            # Soltou antes dos 3 segundos: cancela o progresso
            tempo_inicio_pinca = None

    else:
        # Menu já está aberto e fixo: uma nova pinça (toque rápido) fecha ele
        if pinca_fechada and not pinca_fechada_anterior:
            menu_aberto = False
            menu_pos = None
            tempo_inicio_pinca = None

    pinca_fechada_anterior = pinca_fechada


    # ---- Desenha o menu, se estiver aberto (posição travada) ----
    if menu_aberto and menu_pos is not None:

        x1_menu, y1_menu, x2_menu, y2_menu = menu_pos

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

        cv2.putText(
            frame,
            "(toque a pinca p/ fechar)",
            (x1_menu + 20, y1_menu + 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1
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