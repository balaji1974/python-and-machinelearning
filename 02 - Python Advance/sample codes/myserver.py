# Filename: myserver.py
# Description: A simple TCP server that listens for incoming connections and echoes back messages.
from socket import *
from time import ctime
from threading import Thread

class ClientHandler(Thread):
    def __init__(self, client_socket, client_address):
        super().__init__()
        self.client_socket = client_socket
        self.client_address = client_address

    def run(self):
        print(f"Connection from {self.client_address} has been established.")
        while True:
            data = self.client_socket.recv(1024)
            if not data:
                break
            print(f"Received from {self.client_address}: {data.decode()}")
            response = f"[{ctime()}] {data.decode()}"
            self.client_socket.send(response.encode())
        self.client_socket.close()
        print(f"Connection from {self.client_address} has been closed.")

