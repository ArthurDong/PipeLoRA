
#####################################################################################################
#                                     Load Model Configurature                                      #
#####################################################################################################
import json
from transformers.configuration_utils import PretrainedConfig

class DeepseekConfig(PretrainedConfig):
    def __init__(self, config_filename='./datasets/deepseek_config.json',**kwargs):
        with open(config_filename) as f:
            config = json.load(f)
    
        for key, value in config.items():
            setattr(self, key, value)

        super().__init__(
            **kwargs,
        )