"""Generate speech from text with ElevenLabs."""
import argparse
import os
import sys

from elevenlabs.client import ElevenLabs

DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # "George" premade voice
DEFAULT_MODEL = "eleven_multilingual_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to speak")
    parser.add_argument("-o", "--output", default="output.mp3")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help="voice id")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Set ELEVENLABS_API_KEY first.")
    client = ElevenLabs(api_key=api_key)

    if args.list_voices:
        for voice in client.voices.get_all().voices:
            print(f"{voice.voice_id}  {voice.name}")
        return

    if not args.text:
        parser.error("text is required unless --list-voices is given")

    audio = client.text_to_speech.convert(
        text=args.text,
        voice_id=args.voice,
        model_id=args.model,
        output_format="mp3_44100_128",
    )
    with open(args.output, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
