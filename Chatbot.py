import gradio as gr
import openai
from google.cloud import texttospeech
from langdetect import detect
import io
import base64

# Set your OpenAI API key
openai.api_key = "Your Open-AI Secret Key"

# Initialize conversation with a system message
conversation = [{"role": "system", "content": "You are a human companion and advisor for life and work"}]

def transcribe(audio_file, target_language="en"):
    try:
        # Check if audio_file is None
        if audio_file is None:
            return "No audio file provided."

        # Extract audio data from the tuple
        sample_rate, audio_data = audio_file

        # Convert audio data to base64-encoded string
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')

        # Set a default value for transcript
        transcript = ""

        # comment the following lines if you want to proceed with API calls
        # Whisper API: Transcribe the user's voice
        response = openai.Completion.create(
            engine="whisper-1",
            audio_input={"data": audio_base64},
            prompt="Transcribe the following audio: "
        )
        transcript = response["choices"][0]["text"]
        conversation.append({"role": "user", "content": transcript})

        # ChatGPT API: Generate a response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=transcript,
            temperature=0.7
        )
        system_message = response["choices"][0]["text"]
        conversation.append({"role": "assistant", "content": system_message})

        # Language detection
        detected_lang = detect(transcript)

        # Define a dictionary to map the detected language to language code and voice name
        language_dict = {
            "fr": ("fr-FR", "fr-FR-Standard-A"),
            "es": ("es-ES", "es-ES-Standard-A"),
            "de": ("de-DE", "de-DE-Standard-A"),
            "it": ("it-IT", "it-IT-Standard-A"),
            "zh": ("cmn-CN", "cmn-CN-Standard-A"),
            "ja": ("ja-JP", "ja-JP-Standard-A"),
            # Add more languages as needed
        }

        # Set the language and voice for Google TTS based on the selected language
        if target_language in language_dict:
            language_code, voice_name = language_dict[target_language]
        else:
            # Handle the case where the language is not in the dictionary
            language_code = "en-US"
            voice_name = "en-US-Standard-D"

        # Google Text-to-Speech API: Convert text to speech
        client = texttospeech.TextToSpeechClient.from_service_account_file("C:\\Users\\ASUS\\Downloads\\able-river-405917-a41fa760916a.json")
        input_text = texttospeech.SynthesisInput(text=system_message)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        tts_response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        # Save the audio content to a file
        audio_file_path = f"output_audio_{target_language.lower()}.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(tts_response.audio_content)

        return audio_file_path

    except openai.error.RateLimitError as e:
        # Handle rate limit error
        return f"Rate limit exceeded. Please check your OpenAI plan and billing details. Error: {str(e)}"
    except Exception as e:
        # Handle other exceptions
        return f"An error occurred: {str(e)}"

# Create Gradio Interface with language dropdown
language_dropdown = gr.Dropdown(choices=["en", "fr", "es", "de", "it", "zh", "ja"], label="Choose Language")
bot = gr.Interface(fn=transcribe, inputs=[gr.Audio(type="numpy"), language_dropdown], outputs=gr.Audio(type="filepath"))

# Launch the interface with share=True for a public link
bot.launch(share=True, debug=True)
