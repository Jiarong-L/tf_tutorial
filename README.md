# tf_tutorial


GPU:RTX1650  
CUDA:10.0.130  
cuDNN:7.6.0  
tensorflow2.0  




from keras import backend as k  
from tensorflow.compat.v1 import ConfigProto  
from tensorflow.compat.v1 import InteractiveSession  
config = ConfigProto()  
config.gpu_options.allow_growth = True  
session = InteractiveSession(config=config)  
k.set_session(session)
