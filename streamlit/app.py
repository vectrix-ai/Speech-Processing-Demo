import streamlit as st
from audio_recorder_streamlit import audio_recorder

from IPython.display import Audio
from openai import OpenAI
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
import numpy as np
import uuid, os

client = OpenAI()

st.title('Speech Processing Demo')


MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB

audio_bytes = audio_recorder()
uploaded_file = st.file_uploader("Or upload an audio file", type=['wav'])

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The file size exceeds the limit of 15MB. Please upload a smaller file.")
    else:
        audio_bytes = uploaded_file.read()
    
st.audio(audio_bytes, format="audio/wav")
    

    

prompt = '''Task: Gelieve het bijgevoegde transcript te lezen en volgende drie tabellen aan te vullen.
- Aanwizige compressors
- Het energiecontract
- Mobiliteit

De tabellen mogen in markdown formaat worden aangeleverd.'''

st.subheader('Gemini Model Instructions')
# Add a text input widget
model_instruction= st.text_area("Enter the model instructions here", prompt, height=200)

try:
    # Add a process audio button
    if st.button('Process Audio'):
        # Add a loading spinner
        with st.status('Processing audio file...', expanded=True):
            st.write('Transcribing audio using Whisper ....')
            audio_file_path = f'./tmp/{uuid.uuid4()}.wav'
            with open(audio_file_path, 'wb') as f:
                f.write(audio_bytes)
            audio_file= open(audio_file_path, "rb")
            transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text"
            )

            print('Removing audio file...')
            os.remove(audio_file_path)

            st.write('Summerizing audio using Gemini ...')
            llm_input = "Input: " + transcript + "\n\n" + model_instruction

            client = OpenAI()

            model_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": llm_input,
                    }
                ],
                model="gpt-4",
            )

            #gemini_pro_model = GenerativeModel("gemini-pro")
            #model_response = gemini_pro_model.generate_content(llm_input)

            st.write("Running Text-to-Speech pipeline ...")
        
            
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=model_response.choices[0].message.content
            )

            audio_file = f"./tmp/{uuid.uuid4()}.mp3"
            response.stream_to_file(audio_file)

        st.subheader('Audio Transcript')
        st.success(transcript)

        st.subheader('Gemini Output')
        st.success(model_response.choices[0].message.content)
        st.subheader('Text To Speech Output')   
        st.audio(audio_file, format="audio/mp3")
        os.remove(audio_file)

        # Remove the audio_bytes variable
        del audio_bytes

except Exception as e:
    print(e)
    st.error("Please upload an audio file.")

                     






    








