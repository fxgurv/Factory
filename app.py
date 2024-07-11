import os  # Import the os module for interacting with the operating system
import json  # Import the json module for working with JSON data
import requests  # Import the requests module for making HTTP requests
from g4f.client import Client  # Import the Client class from the g4f.client module
from google.colab import userdata  # Import the userdata module from google.colab
import google.generativeai as genai  # Import the generativeai module from google

from moviepy.editor import (  # Import various classes and functions from moviepy.editor
    VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip, 
    concatenate_videoclips, vfx
)

from elevenlabs import play, save, stream  # Import functions from elevenlabs
from elevenlabs.client import ElevenLabs  # Import the ElevenLabs client

# Define the directory names
Music_dir = "Music"  # Directory for storing music files
Fonts_dir = "Fonts"  # Directory for storing font files
Temp_dir = "Temp"  # Directory for storing temporary files

# Check if the directories exist, if not, create them
if not os.path.exists(Music_dir):
    os.makedirs(Music_dir)  # Create the Music directory
if not os.path.exists(Fonts_dir):
    os.makedirs(Fonts_dir)  # Create the Fonts directory
if not os.path.exists(Temp_dir):
    os.makedirs(Temp_dir)  # Create the Temp directory

# Print a message to confirm the directories were created
print("Directories created successfully!")

# Define parameters for generating content
NICHE = "Space"  # The niche or topic for the content
HEADINGS = "5"  # Number of headings to generate
LANGUAGE = "English"  # Language for the generated content
LLM = "G4f"  # Language model to use (Gimini or G4f)

# Create a prompt for generating headings
prompt = f"Generate {HEADINGS} highlighted headings based video {NICHE}. Your response must be in {LANGUAGE}:"

# Check which language model to use
if LLM == "Gimini":
    # Retrieve the API key for Google Gemini
    api_key = userdata.get('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Please enter your GOOGLE_API_KEY: ")  # Prompt the user for the API key if not found
    genai.configure(api_key=api_key)  # Configure the generative AI with the API key
    model = genai.GenerativeModel('gemini-pro')  # Create a generative model instance
    response = model.generate_content(prompt)  # Generate content using the model
    print(response.text)  # Print the generated response
    
    # Extract headings from the response (assuming it returns a list of headings)
    headings = response.text.split('\n')  # Split the response text into a list of headings
    
else:
    # Use the G4f client to generate content
    client = Client()
    print(f"API call prompt: {prompt}")  # Print the prompt for debugging
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the model to use
        messages=[{"role": "user", "content": prompt}],  # Provide the prompt as a message
        max_tokens=100,  # Maximum number of tokens in the response
        n=5  # Number of responses to generate
    )
    # Extract headings from the response
    headings = [choice.message.content for choice in response.choices]
    for i, heading in enumerate(headings, 1):
        print(f"Heading {i}: {heading}")  # Print each heading

# Process the headings to extract the main parts
list = []
for i in range(len(headings)):
    heading_req = headings[i]
    parts = heading_req.split('. ', 1)
    if len(parts) > 1:
        part_1 = parts[1].split('\n')
        list.append(part_1[0])
print(list)  # Print the processed list of headings

# Create a story prompt using the NICHE
story_prompt = f"Write an engaging one-paragraph short story about {NICHE} for a captivating YouTube Short. [ENGLISH ONLY]"

if LLM == "Gimini":
    # Retrieve the API key for Google Gemini
    api_key = userdata.get('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("Please enter your GOOGLE_API_KEY: ")  # Prompt the user for the API key if not found
    genai.configure(api_key=api_key)  # Configure the generative AI with the API key
    model = genai.GenerativeModel('gemini-pro')  # Create a generative model instance
    response_story = model.generate_content(story_prompt)  # Generate content using the model
    story = response_story.text  # Extract the story from the response

else:
    # Use the G4f client to generate a story
    client = Client()
    response_story = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the model to use
        messages=[{"role": "user", "content": story_prompt}],  # Provide the prompt as a message
        max_tokens=100  # Maximum number of tokens in the response
    )
    story = response_story.choices[0].message.content  # Extract the story from the response

# Print the generated story
print("Generated Story:")
print(story)

# Split the generated story into a list of sentences
sentences = story.split(". ")

# Print the list of sentences
print("\nSentences:")
for sentence in sentences:
    print(sentence)

# Enhance each sentence with additional keywords
enhance = "High Quality, 8k Resolution, Professional, Super Graphics"
enhanced_prompts = [sentence + " " + enhance for sentence in sentences]

# Print the enhanced prompts
print("\nEnhanced Prompts:")
for enhanced_prompt in enhanced_prompts:
    print(enhanced_prompt)

# Choose your model and generate images with enhanced prompts
IMAGE_AI = "lexica"  # Model to use for generating images

# Function to generate an image based on a prompt
def generate_image(prompt: str, index: int) -> str:
    url = f"https://hercai.onrender.com/{IMAGE_AI}/text2image?prompt={prompt}"  # Construct the URL for the API request
    r = requests.get(url)  # Make the API request

    # Check the response status code
    if r.status_code == 200:
        try:
            # Attempt to decode JSON response
            response_data = r.json()
            image_url = response_data.get("url")  # Extract the image URL from the response
            if image_url:
                image_path = os.path.join('Temp', f"image_{index + 1}.jpg")  # Construct the path for saving the image
                with open(image_path, "wb") as image_file:
                    image_file.write(requests.get(image_url).content)  # Save the image content to a file
                return image_path  # Return the path to the saved image
            else:
                print("Error: 'url' key not found in JSON response.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw response: {r.text}")
            return None
    else:
        print(f"Error: Request failed with status code {r.status_code}")
        print(f"Raw response: {r.text}")
        return None

# Generate images for each enhanced prompt
generated_image = [generate_image(prompt, i) for i, prompt in enumerate(enhanced_prompts)]
print(f"Generated Images: {generated_image}")  # Print the list of generated images

# Define the language and TTS settings
TTS = "ElevenLabs"  # Text-to-Speech service to use
VOICE = "Antoni"  # Voice to use for TTS

# Initialize the ElevenLabs client
client = ElevenLabs(
    api_key="sk_7428c5b1dd4c4c0b2f81e7fa795281d85f380190bed4e30d",  # API key for ElevenLabs
)

# Generate the audio using the specified voice and model
for i in range(len(sentences)):
    audio = client.generate(
        text=sentences[i],  # Use sentences[i] to access the text for each iteration
        voice=VOICE,  # Specify the voice to use
        model="eleven_multilingual_v2"  # Specify the model to use
    )

    # Construct the full file path using os.path.join
    file_path = os.path.join(Temp_dir, f"Voice_{i+1}.mp3")

    # Save and play the generated audio
    save(audio, file_path)  # Pass the full file path to the save function

import os  # Import the os module for interacting with the operating system
from moviepy.editor import *  # Import all functions and classes from moviepy.editor

# Define a function to resize the video with a zoom-in effect
def resize_func_1(t):
    return 1 + 0.2 * t  # Zoom-in effect: scale increases over time

# Define a function to resize the video with a stay effect
def resize_func_2(t):
    return 1 + 0.2 * t  # Stay effect: scale remains constant

# Define a function to resize the video with a zoom-out effect
def resize_func_3(t):
    return 1 + 0.2 * (1 - t)  # Zoom-out effect: scale decreases over time


import os  # Import the os module for interacting with the operating system
from moviepy.editor import (  # Import specific classes and functions from moviepy.editor
    VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip, 
    concatenate_videoclips, vfx, ImageClip  # Import ImageClip for handling images
)

# Define the input and output folders
input_folder = 'Temp'  # Folder containing input images
output_folder = 'Temp'  # Folder for saving output videos
audio_folder = 'Temp'  # Folder containing audio files

# Loop through each generated image
for i in range(len(generated_image)):
    input_image = os.path.join(input_folder, f'image_{i+1}.jpg')  # Construct the path to the input image
    image = ImageClip(input_image)  # Load the image as an ImageClip

    # Set a duration for the image clip (e.g., 5 seconds)
    image = image.set_duration(5)

    # Load the audio
    audio_path = os.path.join(audio_folder, f'Voice_{i+1}.mp3')  # Construct the path to the audio file
    audio = AudioFileClip(audio_path)  # Load the audio file

    # Split the audio duration into two halves
    audio_duration = audio.duration
    half_audio_duration = audio_duration / 3
    print(audio_duration)
    print(half_audio_duration)

    # Add the left-to-right transition for the first half of audio duration
    if half_audio_duration > 0:
        transition_duration = 1  # Duration of the transition
        start_position = (0, image.h / 2)  # Starting position on the left
        end_position = (image.w, image.h / 2)  # Ending position on the right

        # Create a copy of the image and set its position to start_position
        left_to_right_clip = image.copy().set_position(start_position)

        # Animate the position to move from start_position to end_position
        left_to_right_clip = left_to_right_clip.set_position(lambda t: (
            start_position[0] + (end_position[0] - start_position[0]) * t,
            start_position[1] + (end_position[1] - start_position[1]) * t
        ))

        left_to_right_clip = left_to_right_clip.set_duration(half_audio_duration)

    if half_audio_duration < audio_duration:
        clip_img = (
            left_to_right_clip
            .set_position(('center', 'center'))
            .set_duration(half_audio_duration / 2)
            .set_fps(10)
        )
        clip_img_1 = (
            clip_img
            .resize(resize_func_2)
            .set_position(('center', 'center'))
            .set_duration(half_audio_duration / 2)
            .set_fps(25)
        )
        clip_img_2 = (
            clip_img_1  # Use the same clip_img_1 as a base
            .resize(resize_func_3)  # Apply zoom-out resizing function
            .set_duration(half_audio_duration)  # Keep the same duration
        )

    # Concatenate the video clips with transitions
    if half_audio_duration > 0 and half_audio_duration < audio_duration:
        final_video = CompositeVideoClip([left_to_right_clip, clip_img, clip_img_1, clip_img_2]).set_audio(audio)
    else:
        final_video = image.set_audio(audio)

    # Export the final video
    output_path = os.path.join(output_folder, f'combined_video_with_transitions{i+1}.mp4')
    final_video.write_videofile(output_path, codec='libx264')

    print("Video with transitions saved to", output_path)

import os  # Import the os module for interacting with the operating system
from moviepy.editor import *  # Import all functions and classes from moviepy.editor

# Initialize an empty list to store video paths
video_paths = []
output_folder = "Temp"  # Define the output folder

# Loop through each generated image
for i in range(len(generated_image)):
    output_path = os.path.join(output_folder, f'combined_video_with_transitions{i+1}.mp4')  # Construct the path to the output video
    video_paths.append(output_path)  # Append the output path to the list of video paths

# Load the video clips
video_clips = [VideoFileClip(video_path) for video_path in video_paths]  # Create a list of VideoFileClip objects from the video paths

# Concatenate the video clips into a single video
final_video = concatenate_videoclips(video_clips, method="compose")

# Define the output path for the final combined video
output_path = os.path.join(output_folder, 'combined_video.mp4')

# Export the final combined video
final_video.write_videofile(output_path, codec='libx264')    
