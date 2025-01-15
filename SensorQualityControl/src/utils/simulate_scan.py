import os
import shutil
import time
import numpy as np

os.chdir("c:/Users/Alexander J. Ross/Documents/QATCH Work/SensorQC/dev/SensorQC/SensorQualityControl/src/utils/")

COPY_FROM = os.path.join(os.getcwd(), "../../content/images/df_c")
COPY_TO = os.path.join(os.getcwd(), "../../content/images/df_c2")
COPY_SPEED = 1  # seconds, can be a 'float' for < 1 sec resolution

if not os.path.exists(COPY_FROM):
    print(f"ERROR: Cannot find directory: {os.path.basename(COPY_FROM)}")
    os.listdir(COPY_FROM)  # will raise 'FileNotFoundError'
if not os.path.exists(COPY_TO):
    print(f"WARNING: Creating missing directory: {os.path.basename(COPY_TO)}")
    os.makedirs(COPY_TO)  # may raise OSError if exists

image_paths = [path for path in os.listdir(COPY_FROM) if path.endswith(
    "jpg") and not path.startswith("stitched")]
image_ids = [int(path[path.rindex("_")+1:-4]) for path in image_paths]
sort_order = np.argsort(image_ids)
sorted_paths = np.array(image_paths)[sort_order]
image_ids = np.array(image_ids)[sort_order]
image_paths = np.array(image_paths)[sort_order]

try:
    for copy_path in image_paths:
        copy_src = os.path.join(COPY_FROM, copy_path)
        copy_dst = os.path.join(COPY_TO, copy_path)
        time.sleep(COPY_SPEED)
        print(
            f"Simulate scan: '{copy_path}' (out of {len(image_paths)} tiles)")
        shutil.copyfile(copy_src, copy_dst)

    print("Finished scanning all tiles!")

except KeyboardInterrupt:
    print("Stopping on user abort...")
    for file in os.listdir(COPY_TO):
        print(f"Deleting '{file}'")
        os.remove(os.path.join(COPY_TO, file))
    print(f"Deleting folder '{os.path.basename(COPY_TO)}'")
    os.removedirs(COPY_TO)
