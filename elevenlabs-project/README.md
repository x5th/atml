# elevenlabs-project

Text-to-speech with the [ElevenLabs](https://elevenlabs.io) API.

## Setup

```bash
pip install -r requirements.txt
export ELEVENLABS_API_KEY=...   # never commit this
```

## Usage

```bash
python tts.py "Hello from ElevenLabs" -o hello.mp3
python tts.py --list-voices
```
