import cv2
import numpy as np
import collections
import time
import os

# Carregando o modelo de detecção (Caffe Model)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Classe que pretendemos detectar
PERSON_CLASS = "person"

# Lista com os endereços RTSP das câmeras IP instaladas no sítio da minha avó
rtsp_urls = [
    "rtsp://username:password@ip_camera1:porta/caminho_stream1",
    "rtsp://username:password@ip_camera2:porta/caminho_stream2",
    "rtsp://username:password@ip_camera3:porta/caminho_stream3"
    # Adicionar ou remover os endereços de câmeras conforme necessário
]

# Duração do buffer
buffer_duration = 4 #em segundos
frame_rate = 20  # Taxa de quadros aproximada (ajustar conforme a qualidade de suas cameras)

# Caminho na rede onde os vídeos serão salvos
network_path = r"\\caminho\para\computador\na\rede"

# Função para salvar o vídeo quando uma pessoa é vista
def save_video(frames, camera_id):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"camera_{camera_id}_detected_{timestamp}.avi"
    filepath = os.path.join(network_path, filename)  # Salva o arquivo no caminho de rede
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, frame_rate, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Vídeo salvo com sucesso: {filepath}")

# Função para processar o feed de vídeo de cada câmera
def process_camera(rtsp_url, camera_id):
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Não foi possível acessar a câmera {camera_id}. Verifique a conexão.")
        return

    # Buffer que armazena os últimos 4 segundos de vídeo
    buffer = collections.deque(maxlen=frame_rate * buffer_duration)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Erro ao capturar frame da câmera {camera_id}.")
            break

        buffer.append(frame)  # Armazena o frame no buffer (para possivelmente ser usado caso necessário)

        # Pré-processamento
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Verifica se alguma detecção corresponde a uma pessoa
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Apenas detecções confiáveis
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # O índice 15 no modelo Caffe corresponde à classe "person"
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = "{}: {:.2f}%".format(PERSON_CLASS, confidence * 100)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Salva os últimos 4 segundos de vídeo ao detectar uma pessoa
                    save_video(list(buffer), camera_id)

        # Exibe o trecho com as detecções
        cv2.imshow(f"Camera {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Processa todas as câmeras definidas na lista
for idx, rtsp_url in enumerate(rtsp_urls):
    process_camera(rtsp_url, idx + 1)
