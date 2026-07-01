import requests

def get_oculus_reading(timeout=0.1):
    url = f"http://127.0.0.1:8000/oculus/data"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        return data
    except Exception as e:
        print(f"Failed to get controller data due to: {e}")
        return None
