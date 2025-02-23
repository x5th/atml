import os
import warnings
import logging
from binance.client import Client


def get_binance_client():
    api_key = os.environ.get("BINANCE_API_KEY")
    secret_key = os.environ.get("BINANCE_SECRET_KEY")
    # Use the live environment by not passing testnet=True (or explicitly set testnet=False)
    return Client(api_key, secret_key)

# Suppress Warnings
warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s'
)

