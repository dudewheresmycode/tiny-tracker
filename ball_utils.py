import os
import sys
import json
import requests


def makeHTTPRequest(httpRequestUrl, eventData):
    try:
        debugLog("httpRequestUrl:", httpRequestUrl)
        debugLog("eventData:", eventData)
        res = requests.post(httpRequestUrl, json=eventData)
        res.raise_for_status()
        # Convert response data to json
        returned_data = res.json()

        debugLog(returned_data)
        result = returned_data['result']
        debugLog("HTTP Response:", result)

    except requests.exceptions.HTTPError as e:  # This is the correct syntax
        debugLog(e)
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        debugLog(e)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def sendEvent(*a):
    print(json.dumps(*a), file=sys.stdout)
    sys.stdout.flush()

def debugLog(*a):
    print(*a, file=sys.stderr)
    sys.stderr.flush()