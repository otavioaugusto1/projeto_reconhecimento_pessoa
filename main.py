import cv2
import numpy as np
import collections
import time

# Carregando o modelo de detecção pré-treinado (Caffe Model)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Classe que queremos detectar
PERSON_CLASS = "person"

# Lista com os endereços RTSP das câmeras IP instaladas na chácara
rtsp_urls = [
    "rtsp://username:password@ip_camera1:porta/caminho_stream1",
    "rtsp://username:password@ip_camera2:porta/caminho_stream2",
    "rtsp://username:password@ip_camera3:porta/caminho_stream3"
    # Adicione mais endereços de câmeras conforme necessário
]

# Buffer para armazenar 60 segundos de vídeo
buffer_duration = 60
frame_rate = 20  # Taxa de quadros aproximada (altere dependendo da qualidade  de suas cameras)

# Função para salvar o vídeo quando uma pessoa é detectada
def save_video(frames, camera_id):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"camera_{camera_id}_detected_{timestamp}.avi"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, frame_rate, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Vídeo salvo com sucesso: {filename}")

# Função para processar o feed de vídeo de cada câmera
def process_camera(rtsp_url, camera_id):
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Não foi possível acessar a câmera {camera_id}. Verifique a conexão.")
        return

    # Buffer para armazenar os últimos 60 segundos de vídeo
    buffer = collections.deque(maxlen=frame_rate * buffer_duration)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Erro ao capturar frame da câmera {camera_id}.")
            break

        buffer.append(frame)  # Armazena o frame no buffer para possível salvamento futuro

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Verifica se alguma detecção corresponde a uma pessoa
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Apenas detecções confiáveis
                idx = int(detections[0, 0, i, 1])
                if idx == 15: 
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = "{}: {:.2f}%".format(PERSON_CLASS, confidence * 100)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Salva os últimos 60 segundos de vídeo quando possivelmente acha uma pessoa
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
