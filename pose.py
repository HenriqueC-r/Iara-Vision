import cv2
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================
# MODELO DAS MÃOS
# ==========================

modelo_mao = os.path.join(
    os.path.dirname(__file__),
    "hand_landmarker.task"
)

base_mao = python.BaseOptions(
    model_asset_path=modelo_mao
)

opcoes_mao = vision.HandLandmarkerOptions(
    base_options=base_mao,
    num_hands=2
)

detector_mao = vision.HandLandmarker.create_from_options(
    opcoes_mao
)


# ==========================
# MODELO DO CORPO
# ==========================

modelo_corpo = os.path.join(
    os.path.dirname(__file__),
    "pose_landmarker.task"
)

base_corpo = python.BaseOptions(
    model_asset_path=modelo_corpo
)

opcoes_corpo = vision.PoseLandmarkerOptions(
    base_options=base_corpo
)

detector_pose = vision.PoseLandmarker.create_from_options(
    opcoes_corpo
)


# ==========================
# CAMERA
# ==========================

camera = cv2.VideoCapture(0)


# Ligações da mão
conexoes_mao = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# Ligações do corpo
conexoes_corpo = [
    (11,12),
    (11,13),
    (13,15),
    (12,14),
    (14,16),

    (11,23),
    (12,24),

    (23,24),

    (23,25),
    (25,27),

    (24,26),
    (26,28)
]


while True:

    sucesso, frame = camera.read()

    if not sucesso:
        break


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ==========================
    # DETECTA MÃOS
    # ==========================

    resultado_mao = detector_mao.detect(mp_image)


    if resultado_mao.hand_landmarks:

        for mao in resultado_mao.hand_landmarks:


            pontos = []

            for ponto in mao:

                x = int(ponto.x * frame.shape[1])
                y = int(ponto.y * frame.shape[0])

                pontos.append((x,y))

                cv2.circle(
                    frame,
                    (x,y),
                    5,
                    (0,255,0),
                    -1
                )


            for inicio,fim in conexoes_mao:

                cv2.line(
                    frame,
                    pontos[inicio],
                    pontos[fim],
                    (255,0,0),
                    2
                )


    # ==========================
    # DETECTA CORPO
    # ==========================

    resultado_pose = detector_pose.detect(mp_image)


    if resultado_pose.pose_landmarks:

        for corpo in resultado_pose.pose_landmarks:

            pontos=[]

            for ponto in corpo:

                x=int(
                    ponto.x * frame.shape[1]
                )

                y=int(
                    ponto.y * frame.shape[0]
                )

                pontos.append((x,y))

                cv2.circle(
                    frame,
                    (x,y),
                    6,
                    (0,0,255),
                    -1
                )


            for inicio,fim in conexoes_corpo:

                if inicio < len(pontos) and fim < len(pontos):

                    cv2.line(
                        frame,
                        pontos[inicio],
                        pontos[fim],
                        (255,255,0),
                        3
                    )


    cv2.putText(
        frame,
        "IARA VISION - Corpo + Maos",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )


    cv2.imshow(
        "IARA Vision",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



camera.release()
cv2.destroyAllWindows()