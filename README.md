# tf_tutorial

### pre-requirement
GPU:RTX1650  
CUDA:10.0.130  
cuDNN:7.6.0  

### in this order add the last 2 lines to PATH:  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\lib64  
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\cudnn\bin  


### install tensorflow-gpu2.0  
python -m pip install --upgrade pip  
pip install --upgrade setuptools  
pip install tensorflow-gpu  

### if use conv2D
from keras import backend as k  
from tensorflow.compat.v1 import ConfigProto  
from tensorflow.compat.v1 import InteractiveSession  
config = ConfigProto()  
config.gpu_options.allow_growth = True  
session = InteractiveSession(config=config)  
k.set_session(session)
### or
gpu = tf.config.experimental.list_physical_devices('GPU')  
tf.config.experimental.set_memory_growth(gpu[0], True)

### or
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3999)])
