from TTS.api import TTS
import whisper
from tempfile import NamedTemporaryFile
import streamlit as st
# from faster_whisper import WhisperModel
from googletrans import Translator






audio = st.file_uploader("Upload an audio file", type=["wav"])

if audio is not None:

    
    with NamedTemporaryFile(delete=False, suffix="wav") as temp_audio:
        
        temp_audio.write(audio.read())
        temp_audio_path = temp_audio.name

    

    st.write(f"Temporary audio file saved at: {temp_audio_path}")

        
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    model = whisper.load_model("base")
    result = model.transcribe(temp_audio_path)
    print(result["text"])
    translator = Translator()
    conv=translator.translate(result["text"], dest='hi')
    print(conv)
    st.write(conv.pronunciation)

    


        
    # tts.tts_to_file(
    #         text="What should i say, All is fine.",
    #         file_path="output.wav",
    #         speaker_wav=temp_audio_path,
    #         language="en"
    #     )

    # st.audio("output.wav", format="audio/wav")
