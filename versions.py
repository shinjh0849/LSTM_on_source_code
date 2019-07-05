# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow as tf
print('tensorflow: %s' % tf.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)

device_name = tf.test.gpu_device_name()
print('@@@@@@ device_name:', device_name)
#if device_name != '\device:GPU:0':
#    raise SystemError('GPU device not found')
#print ('Found GPU at: {}'.format(device_name))
