import socket
import serial


def handle_serial_command(serial_port, baud_rate, command):
    """
    Sends a command to the serial port.
    """
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        ser.write(command.encode())
        ser.close()
        return "Command sent successfully."
    except Exception as e:
        return f"Serial Error: {str(e)}"


def start_server(host, port, serial_port, baud_rate):
    """
    Starts a socket server to listen for commands.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server started on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        data = client_socket.recv(1024).decode()
        if not data:
            break
        print(f"Received: {data}")

        response = handle_serial_command(serial_port, baud_rate, data)
        client_socket.send(response.encode())
        client_socket.close()


if __name__ == "__main__":
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 65432
    SERIAL_PORT = "COM3"  # Adjust this to the correct serial port
    BAUD_RATE = 9600

    start_server(HOST, PORT, SERIAL_PORT, BAUD_RATE)
