"""Synthesizes speech from the input string of text."""
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

# text_client = texttospeech.TextToSpeechClient()

# input_text = texttospeech.SynthesisInput(text="Describe the relationship between wavelength, frequency, and speed of a wave.")
input_text = texttospeech.SynthesisInput(text="Entropy is a measure of disorder or randomness in a system.  The Second Law of Thermodynamics states that the total entropy of an isolated system can only increase over time")

# Note: the voice can also be specified by name.
# Names of voices can be retrieved with client.list_voices().
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Studio-O",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1
)

response = client.synthesize_speech(
    request={"input": input_text, "voice": voice, "audio_config": audio_config}
)


# print("response: ", response.audio_content.split(',')[0])
# The response's audio_content is binary.
# with open("audios/question_10.mp3", "wb") as out:
with open("audios/answer_10.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')