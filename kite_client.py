import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect


def get_kite() -> KiteConnect:
    """
    Creates an authenticated KiteConnect client using .env values.
    Requires:
      KITE_API_KEY
      KITE_ACCESS_TOKEN
    """
    load_dotenv()

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key:
        raise RuntimeError("Missing KITE_API_KEY in .env")
    if not access_token:
        raise RuntimeError("Missing KITE_ACCESS_TOKEN in .env. Run: python kite_login.py")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite
