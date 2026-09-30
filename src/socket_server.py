import socket
import json
import os
import threading
from datetime import datetime


def handle_client(client, addr, log_callback, status_callback, trigger_callback):
    try:
        first_trigger = True
        while True:
            if log_callback:
                if first_trigger:
                    log_callback("Waiting for Trigger")
                else:
                    log_callback("Waiting for Next Trigger")

            raw_data = client.recv(1024)
            if not raw_data:
                break

            data = raw_data.decode().strip()
            if not data:
                continue

            if "TRIGGER" in data.upper():
                first_trigger = False

                if status_callback:
                    status_callback("last_trigger", datetime.now().strftime("%H:%M:%S"))

                if log_callback:
                    log_callback("Trigger Received")
                    log_callback("Detection Started")

                if trigger_callback:
                    response = trigger_callback()
                else:
                    response = "NO_DETECTION"

                if log_callback:
                    log_callback("Detection Finished")

                client.send(response.encode())

                if status_callback:
                    status_callback("last_data", response)

                if log_callback:
                    log_callback("JSON Sent")

    except Exception as e:
        if log_callback:
            log_callback(f"Exception: {e}")
    finally:
        client.close()

        if log_callback:
            log_callback("Client Disconnected")

        if status_callback:
            status_callback("client_ip", "None")


def start_server(log_callback=None, status_callback=None, detection_file="latest_detection.json", trigger_callback=None):

    HOST = "0.0.0.0"
    PORT = 5000

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    server.bind((HOST, PORT))
    server.listen(5)

    if log_callback:
        log_callback("Server Started")
        log_callback(f"Listening on Port {PORT}")

    if status_callback:
        status_callback("server_status", "Running")

    while True:
        try:
            client, addr = server.accept()
            client_ip = addr[0]

            if log_callback:
                log_callback("Client Connected")

            if status_callback:
                status_callback("client_ip", client_ip)

            # Handle the client connection in a separate thread so the main thread
            # can immediately return to server.accept() for the next connection.
            client_thread = threading.Thread(
                target=handle_client,
                args=(client, addr, log_callback, status_callback, trigger_callback),
                daemon=True
            )
            client_thread.start()

        except Exception as e:
            if log_callback:
                log_callback(f"Server Exception: {e}")
