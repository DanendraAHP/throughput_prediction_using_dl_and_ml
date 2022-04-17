import os
class PATH:
    base_dir = os.getcwd() 
    resources = os.path.join(base_dir, 'resources')
    config = os.path.join(base_dir, 'config.yaml')
    model_tf = os.path.join(resources, 'model_tf.h5')
    model_sklearn = os.path.join(resources, 'model_sklearn.sav')