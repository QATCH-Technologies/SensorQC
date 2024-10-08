import json
import os
from datetime import datetime


def log_response(
    response: str, log_folder: str = "logs", log_file: str = "response_log.json"
):
    # Create the logs directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Initialize the data dictionary
    data = {}

    # Split the response into lines
    lines = response.split()
    for line in lines:
        # Split each line by ':' to separate key and value
        key_value = line.split(":")
        key = key_value[0]
        # Handle nested "Count" key
        if key == "Count":
            # Create a nested dictionary for Count
            count_data = {}
            # Process the remaining parts after "Count"
            print(key)
            for count_line in key_value[1:]:
                sub_key, sub_value = count_line.split(":")
                count_data[sub_key] = float(sub_value)  # Convert to float
            data[key] = count_data
        else:
            # Convert other values to float and add to data
            data[key] = float(key_value[1])  # Convert values to float

    # Add timestamp
    data["timestamp"] = datetime.now().isoformat()

    # Log file path
    log_file = os.path.join(log_folder, "response_log.json")

    # Load existing logs
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Append new log entry
    logs.append(data)

    # Save logs back to file
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


if __name__ == "__main__":
    # Example usage
    response = "X:111.50 Y:136.80 Z:63.21 E:0.00 Count X:8920 Y:10944 Z:25284"
    log_response(response)
