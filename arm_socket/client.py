import socket
import time

SERVER_IP = "192.168.10.2"  # Change to server's IP
SERVER_PORT = 5000


def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))

    print("Connected to server.")

    while True:
        message = input("Message to server: ")
        client_socket.sendall(message.encode())

        data = client_socket.recv(1024)
        print(f"Server replies: {data.decode()}")

    client_socket.close()


if __name__ == "__main__":
    main()
