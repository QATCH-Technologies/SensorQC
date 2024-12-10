import socket


def send_serial_command(host, port, command):
    """
    Sends a command to the remote server.
    """
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        client_socket.send(command.encode())
        response = client_socket.recv(1024).decode()
        client_socket.close()
        return response
    except Exception as e:
        return f"Connection Error: {str(e)}"


if __name__ == "__main__":
    HOST = "192.168.1.20"  # IP address of the target laptop
    PORT = 65432
    COMMAND = "Hello, Serial Port!"

    result = send_serial_command(HOST, PORT, COMMAND)
    print("Result:", result)
