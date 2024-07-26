# TinyTracker

TinyTracker is the camera-based putt tracking utility of the TinyGolf game. It was originally built on top of the amazing https://github.com/alleexx/cam-putting-py project, but modified to work as a sub-process of the Unity game.

### Development

Create a python virtual environment:

```bash
python3 -m venv venv
```

Install the basic dependencies:

```bash
./venv/bin/pip install -r requirements.txt
```

To generate a new pyinstaller spec file:

```bash
./venv/bin/pyi-makespec \
--add-data "images/splash.png:images" \
--onefile \
ball_tracking.py
```

> _Note: To update the requirements file after installing any new python packages (`venv/bin/pip install`) run:<br>`./venv/bin/pip freeze > requirements.txt`_

#### Running

To run the app locally:

```bash
./venv/bin/python ball_tracking.py

## pass a custom config
./venv/bin/python ball_tracking.py --config ./config.ini
```

#### Building

```bash
./venv/bin/python -m PyInstaller ball_tracking.spec
```
