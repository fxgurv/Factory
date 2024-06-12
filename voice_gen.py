import os
import elevenlabs

elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
elevenlabs.set_api_key(elevenlabs_api_key)

def generate_voice_over(script):
    voice = elevenlabs.Voice(
        voice_id = "ErXwobaYiN019PkySvjV",
        settings= elevenlabs.VoiceSettings(
            stability=0.5,
            similarity_boost=0.75
        )
    )

    voice_over = elevenlabs.generate(
        text=script, 
        voice=voice)
    elevenlabs.save(voice_over, "voice_over.mp3")
