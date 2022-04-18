from src.common.yaml_util import read_yaml_file
from src.common.constant import PATH

EXPLANATION_TEXT = read_yaml_file(PATH.config)
EXPLANATION_TEXT = EXPLANATION_TEXT['explanation_text']
print(EXPLANATION_TEXT['optimizer'])