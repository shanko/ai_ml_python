import os
import re
from dotenv import load_dotenv
from io import BytesIO
import requests
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

audio = elevenlabs.text_to_speech.convert(
    text="This is an example of using the eleven labs voice AI library written in Python to convert text to speech.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

## TTS
play(audio)


#    "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
#    "https://sample-files.com/downloads/audio/mp3/tone-test.mp3"
audio_url = (
    "https://sample-files.com/downloads/audio/mp3/voice-sample.mp3"
)
response = requests.get(audio_url)
audio_data = BytesIO(response.content)

transcription = elevenlabs.speech_to_text.convert(
    file=audio_data,
    model_id="scribe_v1",  # Model to use, for now only "scribe_v1" is supported
    tag_audio_events=True, # Tag audio events like laughter, applause, etc.
    language_code="eng",   # Language of the audio file. If set to None, the model will detect the language automatically.
    diarize=True,          # Whether to annotate who is speaking
)

# Print only the text for each word in the transcription.words array
txt = ""
for word in transcription.words:
    txt += word.text

## STT
# Remove contiguous whitespace from the txt string
txt = re.sub(r'\s+', ' ', txt).strip()
print(txt)

## TODO:
# https://github.com/NexusRanger/Elevenlabs-Phrase-Recycler
# ^^^ Learn how to upload .mp3 sound file and convert it to text
