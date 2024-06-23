import warnings
from config import get_config
from train import train_model
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("translation.log"),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    try:
        logging.info("Code Start...")
        warnings.filterwarnings('ignore')  # Filtering warnings
        config = get_config()  # Retrieving config settings
        logging.info("Config Loaded Successfully!")
        logging.info("Training Start...")
        train_model(config)  # Training model with the config arguments
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        print(f"An error occurred: {e}")  # Debug print