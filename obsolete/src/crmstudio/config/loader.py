import os
import yaml

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        self.validate_config(config)
        return config

    def validate_config(self, config):
        required_keys = ['models', 'report']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Additional validation logic can be added here