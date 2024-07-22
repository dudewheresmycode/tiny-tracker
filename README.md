## TinyTracker

TinyTracker is the camera based putt tracker component of the TinyPutt game. It was originally built on top of the amazing https://github.com/alleexx/cam-putting-py project, but modified to work as a sub-process of Unity a project.

### Development

Install the basic dependencies:

```bash
pip install -r requirements.txt
# or in a venv (MacOS)
./bin/pip install -r requirements.txt
```

### Compiling

```bash
pyinstaller ball_tracking.spec
# or in a venv (MacOS)
./bin/python -m PyInstaller ball_tracking.spec
```
