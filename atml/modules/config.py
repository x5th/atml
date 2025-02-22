import os
import warnings
import logging
from binance.client import Client

# Suppress Warnings
warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def get_binance_client():
    """Fetches Binance Client using environment variables for API keys.
    Make sure you set:
      export BINANCE_API_KEY='...'
      export BINANCE_SECRET_KEY='...'
    """
    api_key = os.environ.get("BINANCE_API_KEY")
    secret_key = os.environ.get("BINANCE_SECRET_KEY")
    return Client(api_key, secret_key, testnet=True)
