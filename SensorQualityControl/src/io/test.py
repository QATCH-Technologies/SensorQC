import subprocess


def execute_command(command):
    """Executes a system command and prints the output."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Error executing command: {command}")
            print(result.stderr)
    except Exception as e:
        print(f"An exception occurred: {e}")


def disable_device(device_id):
    """Disables the device with the given device ID."""
    print(f"Disabling device: {device_id}")
    command = f'devcon disable "{device_id}"'
    execute_command(command)


def enable_device(device_id):
    """Enables the device with the given device ID."""
    print(f"Enabling device: {device_id}")
    command = f'devcon enable "{device_id}"'
    execute_command(command)


def main():
    device_id = r"USB\VID_0FD9&PID_0094&MI_00\7&2472f605&0&0000"
    disable_device(device_id)
    enable_device(device_id)


if __name__ == "__main__":
    main()
