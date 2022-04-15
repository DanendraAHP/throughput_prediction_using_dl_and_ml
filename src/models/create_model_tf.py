from src.common.yaml_util import read_yaml_file, write_yaml
import streamlit as st

def get_config(file):
    config = read_yaml_file(file)
    model_tf = config['model_tf']
    model_tf_layer = model_tf['layers']
    model_tf_units = model_tf['units']
    return model_tf_layer, model_tf_units

def write_to_yaml(file, model_tf_layer, model_tf_units):
    config = {'model_tf':{'layers':model_tf_layer,'units':model_tf_units}}
    write_yaml(config, file)

def add_layer(file, layer, unit):
    model_tf_layer, model_tf_units = get_config(file)
    if model_tf_layer is None or model_tf_units is None:
        model_tf_layer = []
        model_tf_units =[]
    model_tf_layer.append(layer)
    model_tf_units.append(unit)
    write_to_yaml(file, model_tf_layer,model_tf_units)

def remove_layer(file):
    model_tf_layer, model_tf_units = get_config(file)
    if model_tf_layer is None or model_tf_units is None:
        st.write('No more layer in model')
    model_tf_units.pop()
    model_tf_layer.pop()
    write_to_yaml(file, model_tf_layer,model_tf_units)

def check_layer(file):
    model_tf_layer, model_tf_units = get_config(file)
    for i in range(1, len(model_tf_layer)):
        if model_tf_layer[i] == 'LSTM' and model_tf_layer[i-1] == 'Dense':
            st.write('Model layer error')
            return False
    return True

def clear_layer(file):
    write_to_yaml(file, None, None)