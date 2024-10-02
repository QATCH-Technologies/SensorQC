# Notes on Quality Control Automation

## Project Brief
Build an automated system to detect issues with sensor quality and report issues along with sensor ID to QC team for further testing.

## Hardware Overview
- Stage to move the wafer (maybe repurpose a 3D printer for moving optics around).
- 3D Printer to 4-Axis Camera Mount --> https://www.instructables.com/Turn-Your-Old-3dprinter-Into-a-REMOTE-4-AXIS-CAMER/
- Notches for wafer alignment.
- Lighting for sensor illumination.
- Sensor optics (TBD).

## Software Overview
- System to operate the wafter stage.
- System to operate the sensor optics/lighting.
- Analysis system (i.e. the ML component)

### Dependencies
- TensorFlow
- Pillow
- CV2

### Machine Learning Notes
- Likely end up using TensorFlow CNN for this
- Object detection lens?
- Use something like ResNet YOLO architecture.
- YOLOv7 demo --> https://www.kaggle.com/code/taranmarley/yolo-v7-object-detection
- YOLOv9 example --> https://www.kaggle.com/code/ihsncnkz/face-mask-detection-with-yolov9
- Good algo summary --> https://viso.ai/deep-learning/object-detection/
- Anomaly detection lens
- Classification?
- Maybe look at it a Good/Bad classification layer followed by an object detection layer for picking out issues with a sensor.

#### Target Issues
- Broken Channels
- Collapsed Channels
- Blocked Inlet/Outlet

Example CNN
- Pneumonia Detection model used this setup: 
Model: "sequential_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_56 (Conv2D)           (None, 150, 150, 32)      320       
_________________________________________________________________
batch_normalization_56 (Batc (None, 150, 150, 32)      128       
_________________________________________________________________
max_pooling2d_44 (MaxPooling (None, 75, 75, 32)        0         
_________________________________________________________________
conv2d_57 (Conv2D)           (None, 75, 75, 64)        18496     
_________________________________________________________________
dropout_45 (Dropout)         (None, 75, 75, 64)        0         
_________________________________________________________________
batch_normalization_57 (Batc (None, 75, 75, 64)        256       
_________________________________________________________________
max_pooling2d_45 (MaxPooling (None, 38, 38, 64)        0         
_________________________________________________________________
conv2d_58 (Conv2D)           (None, 38, 38, 64)        36928     
_________________________________________________________________
batch_normalization_58 (Batc (None, 38, 38, 64)        256       
_________________________________________________________________
max_pooling2d_46 (MaxPooling (None, 19, 19, 64)        0         
_________________________________________________________________
conv2d_59 (Conv2D)           (None, 19, 19, 128)       73856     
_________________________________________________________________
dropout_46 (Dropout)         (None, 19, 19, 128)       0         
_________________________________________________________________
batch_normalization_59 (Batc (None, 19, 19, 128)       512       
_________________________________________________________________
max_pooling2d_47 (MaxPooling (None, 10, 10, 128)       0         
_________________________________________________________________
conv2d_60 (Conv2D)           (None, 10, 10, 256)       295168    
_________________________________________________________________
dropout_47 (Dropout)         (None, 10, 10, 256)       0         
_________________________________________________________________
batch_normalization_60 (Batc (None, 10, 10, 256)       1024      
_________________________________________________________________
max_pooling2d_48 (MaxPooling (None, 5, 5, 256)         0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 6400)              0         
_________________________________________________________________
dense_23 (Dense)             (None, 128)               819328    
_________________________________________________________________
dropout_48 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 129

### Annotations
Use labelme for annotations