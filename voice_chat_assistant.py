from openai import OpenAI
import os
import base64
import time
from collections import deque

class CoreFunctions:
    def __init__(self, api_key):
        self.openai_client = OpenAI(api_key=api_key)

    def cleanup_audio_files(self):
        """Clean up any temporary audio files"""
        try:
            for file in os.listdir():
                if (file.startswith("audio_input_") or file.startswith("audio_response_")) and file.endswith(".mp3"):
                    try:
                        os.remove(file)
                    except Exception as e:
                        raise Exception(f"Could not remove audio file {file}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during audio cleanup: {str(e)}")

    def transcribe_audio(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text

    def text_to_audio(self, text, audio_path):
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        response.stream_to_file(audio_path)

    def get_base64_audio(self, audio_file):
        if os.path.exists(audio_file):
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode("utf-8")
        return None

    def call_llm(self, model, messages, temp):
        """Call OpenAI's LLM with streaming"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                stream=True
            )
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")

        full_response = ""

        try:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            return full_response

        except Exception as e:
            raise Exception(f"Error in streaming response: {str(e)}")

    def process_message(self, input_text, conversation_memory, is_voice=False):
        system_message = """You are an AI assistant specialized in having friendly conversations. 
        Please be concise and natural in your responses, as they may be converted to speech."""

        condensed_history = "\n".join(
            [f"Human: {exchange['question']}\nAI: {exchange['answer']}" 
             for exchange in conversation_memory]
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Conversation history:\n{condensed_history}\n\nQuestion: {input_text}"}
        ]

        response = self.call_llm("gpt-4", messages, 0.7)
        
        if is_voice:
            audio_file = f"audio_response_{int(time.time())}.mp3"
            self.text_to_audio(response, audio_file)
            return response, audio_file
        
        return response, None