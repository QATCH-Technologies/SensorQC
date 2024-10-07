import time
import importlib

try:
    DNX64 = getattr(importlib.import_module("DNX64"), "DNX64")
except ImportError as err:
    print("Error: ", err)

# Initialize the DNX64 class
dll_path = "/path/to/DNX64.dll"
micro_scope = DNX64(dll_path)

# Set Device Index first
micro_scope.SetVideoDeviceIndex(0)

# Get total number of video devices being detected
device_count = micro_scope.GetVideoDeviceCount()
print(f"Number of video devices: {device_count}")
# NOTE: Buffer time for devices to set up properly
time.sleep(0.1)

# Set the auto-exposure target value for device 0
micro_scope.SetAETarget(0, 100)
# NOTE: Buffer time for devices to set up properly
time.sleep(0.1)

# Set the exposure value for device 0
micro_scope.SetExposureValue(0, 1000)
