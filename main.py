import flet as ft
import psutil
import requests
import asyncio
import torch
import torchaudio
import os
import onnxruntime as ort
import numpy as np
import warnings
import pyaudio
import wave
import scipy.special
import threading
import webrtcvad
import time

warnings.filterwarnings('ignore')

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30
CHUNK = int(RATE * FRAME_DURATION / 1000)

COMMAND_ACTIONS = {
    "go": "Smarthomefunktion gestartet",
    "stop": "Alle Geräte gestoppt",
    "on": "Licht eingeschaltet",
    "off": "Licht ausgeschaltet",
    "up": "Jalousien gehen hoch",
    "down": "Jalousien gehen runter",
    "left": "Menübewegung links",
    "right": "Menübewegung rechts",
    "yes": "Befehl bestätigt",
    "no": "Befehl abgebrochen",
}


def win(page: ft.Page):
    ram = ft.Text("None", size=12)
    feat_model_name = ft.Text("None", size=12)
    clf_model_name = ft.Text("None", size=12)
    net = ft.Text("None", size=12)
    cpu = ft.Text("None", size=12)
    feat_model = None
    classifier = None
    label_encoder = None
    clf_size_mb = ft.Text('None', size=12)
    feat_extr_size_mb = ft.Text('None', size=12)
    latency = ft.Text("None", size=12)
    whole_model_path = r'C:\GitHub\EdgeAI\classifier_model_12.pt'
    onnx_feat_model_path = r'C:\GitHub\EdgeAI\wav2vec.onnx_12'
    onnx_clf_model_path = r'C:\GitHub\EdgeAI\classifier.onnx_12'
    output_filename = "audio-file.wav"
    pred_path = None
    predicted_label = None
    confidence = None
    p = pyaudio.PyAudio()

    status_display = ft.Container(
        content=ft.Text("Warte auf Start...", size=8, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
        width=400,
        height=50,
        bgcolor=ft.Colors.BLUE_100,
        border_radius=15,
        padding=5,
    )

    marvin_confidence_display = ft.Container(
        content=ft.Column([
            ft.Text("Wakeword-Erkennung", size=8, weight=ft.FontWeight.BOLD),
            ft.Text("---", size=8),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400,
        height=50,
        bgcolor=ft.Colors.GREY_200,
        border_radius=15,
        padding=5,
    )

    command_confidence_display = ft.Container(
        content=ft.Column([
            ft.Text("Kommando-Erkennung", size=8, weight=ft.FontWeight.BOLD),
            ft.Text("---", size=8),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        width=400,
        height=50,
        bgcolor=ft.Colors.GREY_200,
        border_radius=15,
        padding=5,
    )

    action_display = ft.Container(
        content=ft.Text("Keine Aktion", size=8, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
        width=400,
        height=50,
        bgcolor=ft.Colors.GREY_300,
        border_radius=15,
        padding=5,
        animate=ft.Animation(300, "easeOut"),
    )

    model_latency = ft.Container(
        content=ft.Text("Latency:", size=8, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
        width=400,
        height=25,
        bgcolor=ft.Colors.GREY_300,
        border_radius=15,
        padding=5,
        animate=ft.Animation(300, "easeOut"),
    )

    def update_status(text, color):
        status_display.content = ft.Text(text, size=8, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
        status_display.bgcolor = color
        page.update()

    def update_marvin_confidence(label, conf, recognized):
        if recognized:
            bg_color = ft.Colors.GREEN_100
            text_color = ft.Colors.GREEN_900
        else:
            bg_color = ft.Colors.RED_100
            text_color = ft.Colors.RED_900

        marvin_confidence_display.content = ft.Column([
            ft.Text("Wakeword-Erkennung", size=8, weight=ft.FontWeight.BOLD),
            ft.Text(f"{label}: {conf:.1f}%", size=8, color=text_color, weight=ft.FontWeight.BOLD),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        marvin_confidence_display.bgcolor = bg_color
        page.update()

    def update_command_confidence(label, conf, marvin_ok):
        if marvin_ok:
            bg_color = ft.Colors.BLUE_100
            text_color = ft.Colors.BLUE_900
        else:
            bg_color = ft.Colors.GREY_200
            text_color = ft.Colors.GREY_600

        command_confidence_display.content = ft.Column([
            ft.Text("Kommando-Erkennung", size=8, weight=ft.FontWeight.BOLD),
            ft.Text(f"{label}: {conf:.1f}%", size=8, color=text_color, weight=ft.FontWeight.BOLD),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        command_confidence_display.bgcolor = bg_color
        page.update()

    def show_action(action_text):
        action_display.content = ft.Text(action_text, size=8, weight=ft.FontWeight.BOLD,
                                         text_align=ft.TextAlign.CENTER)
        action_display.bgcolor = ft.Colors.GREEN_100
        action_display.scale = 1.05
        page.update()
        action_display.scale = 1.0
        page.update()

    def update_latency(latency_ms):
        model_latency.content = ft.Text(
            f"Latency: {latency_ms:.1f}ms",
            size=8,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER
        )
        page.update()

    def load_models():
        try:
            model_as_dict = torch.load(whole_model_path, weights_only=False)
            onnx_feature_model = ort.InferenceSession(onnx_feat_model_path)
            onnx_clf_model = ort.InferenceSession(onnx_clf_model_path)
            label_encoder = model_as_dict['label_encoder']
            clf_model_name.value = os.path.basename(onnx_clf_model_path)
            clf_model_name.color = ft.Colors.GREEN
            feat_model_name.value = os.path.basename(onnx_feat_model_path)
            feat_model_name.color = ft.Colors.GREEN
            clf_size_mb.value = round((os.path.getsize(onnx_clf_model_path) / 1000000), 1)
            feat_extr_size_mb.value = round((os.path.getsize(onnx_feat_model_path) / 1000000), 1)
            page.update()
            return onnx_feature_model, onnx_clf_model, label_encoder

        except Exception as e:
            clf_model_name.value = "Kein Classifier geladen"
            clf_model_name.color = ft.Colors.RED
            feat_model_name.value = "Kein Feature Extraktor geladen"
            feat_model_name.color = ft.Colors.RED
            print(f'Fehler: {e}')

    def ram_size():
        ram.value = f'{round(psutil.Process().memory_info().rss / 1024 / 1024, 1)} MB'
        page.update()

    def internet_check():
        try:
            requests.get("https://www.google.com", timeout=3)
            return True
        except OSError:
            return False

    def cpu_check():
        cpu.value = f'{psutil.cpu_percent(interval=1)}%'

    async def metric_update():
        while True:
            cpu_check()
            ram_size()
            net.value, net.color = ("online", ft.Colors.GREEN) if internet_check() else ("offline", ft.Colors.RED)
            page.update()
            await asyncio.sleep(1)

    def inference(e):
        waveform, sample_rate = torchaudio.load(pred_path)
        target_length = 16000
        current_length = waveform.shape[1]
        nonlocal predicted_label, confidence

        if current_length > target_length:
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - current_length))

        inference_start = time.perf_counter()
        feat_input_name = feat_model.get_inputs()[0].name
        feat_output = feat_model.run([feat_model.get_outputs()[0].name],
                                     {feat_input_name: waveform.numpy().astype(np.float32)})
        features_aggregated = feat_output[0].mean(axis=1)

        clf_start_time = time.perf_counter()
        clf_input_name = classifier.get_inputs()[0].name
        clf_output = classifier.run([classifier.get_outputs()[0].name], {clf_input_name: features_aggregated})
        logits = clf_output[0]

        prediction_class = np.argmax(logits, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([prediction_class])[0]
        probabilities = scipy.special.softmax(logits[0])
        confidence = probabilities[prediction_class] * 100

        inference_end = time.perf_counter()
        latency.value = inference_end - inference_start
        update_latency(latency.value * 1000)

    def word_check(e):
        nonlocal pred_path, predicted_label, confidence

        update_status("Warte auf 'Marvin'...", ft.Colors.ORANGE_100)
        lauschen_func()
        pred_path = output_filename

        inference(e)
        marvin_label = predicted_label
        marvin_conf = confidence

        marvin_recognized = marvin_label == "marvin" and marvin_conf > 40
        update_marvin_confidence(marvin_label, marvin_conf, marvin_recognized)

        if marvin_recognized:
            update_status("Marvin erkannt - Warte auf Kommando...", ft.Colors.GREEN_100)
            lauschen_func()
            pred_path = output_filename

            inference(e)
            command_label = predicted_label
            command_conf = confidence

            update_command_confidence(command_label, command_conf, True)

            if command_label in COMMAND_ACTIONS:
                show_action(COMMAND_ACTIONS[command_label])
                time.sleep(1.0)
            update_status("Bereit für neues Wake-Word", ft.Colors.BLUE_100)
        else:
            update_status("Kein Marvin erkannt", ft.Colors.RED_100)
            time.sleep(1.0)
            update_status("Bereit für neues Wake-Word", ft.Colors.BLUE_100)

    def lauschen_func():
        vad = webrtcvad.Vad(3)
        audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        STATE_WAITING, STATE_RECORDING = 0, 1
        state = STATE_WAITING
        frames = []
        silence_counter = 0
        SILENCE_THRESHOLD = 15

        try:
            while True:
                frame = audio_stream.read(CHUNK)
                is_speech = vad.is_speech(frame, RATE)

                if state == STATE_WAITING:
                    if is_speech:
                        state = STATE_RECORDING
                        frames = [frame]
                        silence_counter = 0

                elif state == STATE_RECORDING:
                    frames.append(frame)
                    if is_speech:
                        silence_counter = 0
                    else:
                        silence_counter += 1
                        if silence_counter >= SILENCE_THRESHOLD:
                            with wave.open(output_filename, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(p.get_sample_size(FORMAT))
                                wf.setframerate(RATE)
                                wf.writeframes(b''.join(frames))

                            audio_stream.stop_stream()
                            audio_stream.close()
                            return output_filename

        except Exception as e:
            audio_stream.stop_stream()
            audio_stream.close()
            raise e

    def continious_listening(e):
        def listening_thread():
            while True:
                word_check(e)
                time.sleep(0.5)

        threading.Thread(target=listening_thread, daemon=True).start()

    feat_model, classifier, label_encoder = load_models()
    page.run_task(metric_update)

    page.add(
        ft.Column([
            status_display,
            marvin_confidence_display,
            command_confidence_display,
            model_latency,
            action_display,
            ft.Divider(height=20),
            ft.Row([ft.Text("RAM-Nutzung:", size=12), ram]),
            ft.Row([ft.Text("Modell zur feature extraction:", size=12), feat_model_name]),
            ft.Row([ft.Text("Modellgröße des Feature_Extractors:", size=12), feat_extr_size_mb, ft.Text("MB")]),
            ft.Row([ft.Text("Modell zur Klassifizierung:", size=12), clf_model_name]),
            ft.Row([ft.Text("Modellgröße des Classifiers:", size=12), clf_size_mb, ft.Text("MB")]),
            ft.Row([ft.Text("Internet-Status:", size=12), net]),
            ft.Row([ft.Text("CPU-Last:", size=12), cpu]),
            ft.Container(height=20),
            ft.ElevatedButton('Dauerhaft lauschen', on_click=continious_listening, width=400, height=50),
            ft.Container(height=10),
            ft.Text('"Marvin" als Wake-Word', size=14, text_align=ft.TextAlign.CENTER),
            ft.Text('Verfügbare Kommandos: "go", "on", "off", "up", "yes", "no", "down", "left", "right", "stop"',
                    size=16, text_align=ft.TextAlign.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )
    page.on_close = lambda e: p.terminate()


ft.app(win)