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

# Tempo de espera (segundos) segurando a pinça parada pra abrir/fechar o menu
TEMPO_HOLD = 3.0

# Quanto a mão precisa se mover (em % da largura do frame) pra virar arraste
# em vez de contagem de fechamento
LIMIAR_MOVIMENTO_FRAC = 0.03

# Tamanho do painel do menu (em pixels)
MENU_LARGURA = 260
MENU_ALTURA = 220

# Margem de tolerância ao redor do painel pra contar como "pinçou nele"
MARGEM_PAINEL = 25

# Raio do anel de progresso
RAIO_ANEL = 40


# ---- Estado do menu (persiste entre frames) ----
menu_aberto = False
menu_pos = None                # (x1, y1, x2, y2) do painel
modo = "nenhum"                 # "nenhum" | "avaliando" | "arrastando" | "fechando"
tempo_inicio_pinca = None
pinca_inicio_pos = None         # (x, y) de onde a pinça começou, pra medir deslocamento
offset_arraste = None           # (dx, dy) entre a pinça e o canto do painel
pinca_fechada_anterior = False


def desenhar_anel_progresso(frame, cx, cy, progresso):
    """Desenha o anel de carregamento (0.0 a 1.0) ao redor do ponto da pinça."""

    cv2.circle(
        frame,
        (cx, cy),
        RAIO_ANEL,
        (80, 80, 80),
        4
    )

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


def ponto_dentro(px, py, retangulo, margem=0):
    x1, y1, x2, y2 = retangulo
    return (x1 - margem) <= px <= (x2 + margem) and (y1 - margem) <= py <= (y2 + margem)


def clamp_menu(x1, y1, largura_frame, altura_frame):
    """Recebe o canto superior esquerdo desejado e devolve a posição clampada."""

    x2 = x1 + MENU_LARGURA
    y2 = y1 + MENU_ALTURA

    if x1 < 0:
        x1 = 0
        x2 = MENU_LARGURA
    if x2 > largura_frame:
        x2 = largura_frame
        x1 = largura_frame - MENU_LARGURA
    if y1 < 0:
        y1 = 0
        y2 = MENU_ALTURA
    if y2 > altura_frame:
        y2 = altura_frame
        y1 = y2 - MENU_ALTURA

    return (x1, y1, x2, y2)


conexoes = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao acessar câmera")
        break


    altura_frame, largura_frame = frame.shape[:2]
    limiar_movimento_px = LIMIAR_MOVIMENTO_FRAC * largura_frame

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
    mao_detectada = None

    if resultado.hand_landmarks:

        for mao in resultado.hand_landmarks:

            mao_detectada = mao
            polegar = mao[4]
            indicador = mao[8]

            distancia = math.hypot(
                polegar.x - indicador.x,
                polegar.y - indicador.y
            )

            pinca_fechada = distancia < LIMIAR_PINCA

            ancora_x = int(((polegar.x + indicador.x) / 2) * largura_frame)
            ancora_y = int(((polegar.y + indicador.y) / 2) * altura_frame)


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


    rising_edge = pinca_fechada and not pinca_fechada_anterior


    # ---- Máquina de estados do menu ----

    if not menu_aberto:

        modo = "nenhum"

        if pinca_fechada:

            if tempo_inicio_pinca is None:
                tempo_inicio_pinca = time.time()

            decorrido = time.time() - tempo_inicio_pinca
            progresso = min(decorrido / TEMPO_HOLD, 1.0)

            if ancora_x is not None:
                desenhar_anel_progresso(frame, ancora_x, ancora_y, progresso)

            if decorrido >= TEMPO_HOLD:

                x1_menu = ancora_x + 20
                y1_menu = ancora_y - MENU_ALTURA - 20

                menu_pos = clamp_menu(x1_menu, y1_menu, largura_frame, altura_frame)
                menu_aberto = True
                tempo_inicio_pinca = None

        else:
            tempo_inicio_pinca = None

    else:
        # ---- Menu aberto: pode estar parado, sendo avaliado, arrastando ou fechando ----

        if modo == "arrastando":

            if pinca_fechada and ancora_x is not None:

                novo_x1 = ancora_x - offset_arraste[0]
                novo_y1 = ancora_y - offset_arraste[1]
                menu_pos = clamp_menu(novo_x1, novo_y1, largura_frame, altura_frame)

            else:
                # Soltou: trava a posição atual
                modo = "nenhum"
                offset_arraste = None

        elif modo == "avaliando":
            # Pinça começou EM CIMA do painel. Só falta descobrir se você
            # quer arrastar (vai mover a mão) ou fechar (vai ficar parado)

            if pinca_fechada and ancora_x is not None:

                deslocamento = math.hypot(
                    ancora_x - pinca_inicio_pos[0],
                    ancora_y - pinca_inicio_pos[1]
                )

                if deslocamento > limiar_movimento_px:
                    # Começou a mover -> vira arraste
                    modo = "arrastando"

                else:
                    # Continua parado -> conta pro fechamento
                    decorrido = time.time() - tempo_inicio_pinca
                    progresso = min(decorrido / TEMPO_HOLD, 1.0)

                    desenhar_anel_progresso(frame, ancora_x, ancora_y, progresso)

                    if decorrido >= TEMPO_HOLD:
                        menu_aberto = False
                        menu_pos = None
                        modo = "nenhum"
                        tempo_inicio_pinca = None

            else:
                # Soltou antes de decidir: cancela, não faz nada (evita fechar sem querer)
                modo = "nenhum"
                tempo_inicio_pinca = None

        elif modo == "fechando":
            # Pinça começou FORA do painel -> só serve pra fechar

            if pinca_fechada:

                decorrido = time.time() - tempo_inicio_pinca
                progresso = min(decorrido / TEMPO_HOLD, 1.0)

                if ancora_x is not None:
                    desenhar_anel_progresso(frame, ancora_x, ancora_y, progresso)

                if decorrido >= TEMPO_HOLD:
                    menu_aberto = False
                    menu_pos = None
                    modo = "nenhum"
                    tempo_inicio_pinca = None

            else:
                modo = "nenhum"
                tempo_inicio_pinca = None

        else:
            # modo == "nenhum": esperando uma nova pinça decidir o que fazer
            if rising_edge and ancora_x is not None:

                if ponto_dentro(ancora_x, ancora_y, menu_pos, margem=MARGEM_PAINEL):
                    # Pinçou em cima do painel -> ainda não sabemos se é arraste ou fechar
                    modo = "avaliando"
                    pinca_inicio_pos = (ancora_x, ancora_y)
                    tempo_inicio_pinca = time.time()

                    x1_menu, y1_menu, _, _ = menu_pos
                    offset_arraste = (ancora_x - x1_menu, ancora_y - y1_menu)

                else:
                    # Pinçou fora do painel -> só pode ser fechamento
                    modo = "fechando"
                    tempo_inicio_pinca = time.time()

    pinca_fechada_anterior = pinca_fechada


    # ---- Desenha o menu, se estiver aberto ----
    if menu_aberto and menu_pos is not None:

        # Desenha a mão por cima do menu
        if mao_detectada is not None:

            for ponto in mao_detectada:

                x = int(ponto.x * largura_frame)
                y = int(ponto.y * altura_frame)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            for inicio, fim in conexoes:

                x1 = int(mao_detectada[inicio].x * largura_frame)
                y1 = int(mao_detectada[inicio].y * altura_frame)

                x2 = int(mao_detectada[fim].x * largura_frame)
                y2 = int(mao_detectada[fim].y * altura_frame)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


        x1_menu, y1_menu, x2_menu, y2_menu = menu_pos

        cv2.rectangle(
            frame,
            (x1_menu, y1_menu),
            (x2_menu, y2_menu),
            (40, 40, 40),
            -1
        )

        if modo == "arrastando":
            cor_borda = (0, 255, 0)
        elif modo == "avaliando":
            cor_borda = (0, 200, 255)
        else:
            cor_borda = (0, 255, 255)

        cv2.rectangle(
            frame,
            (x1_menu, y1_menu),
            (x2_menu, y2_menu),
            cor_borda,
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
            "pince e mova = arrasta",
            (x1_menu + 20, y1_menu + 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1
        )

        cv2.putText(
            frame,
            "pince e segure 3s = fecha",
            (x1_menu + 20, y1_menu + 195),
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