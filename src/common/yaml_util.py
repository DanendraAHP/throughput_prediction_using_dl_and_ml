import yaml

def read_yaml_file(file):
    with open(file) as read_file:
        try:
            data = yaml.safe_load(read_file)  
        except yaml.YAMLError as exc:
            print(exc)
    return data

def write_yaml(dictionary, file):
    with open(file, 'w') as dump_file:
        outputs = yaml.dump(dictionary, dump_file)