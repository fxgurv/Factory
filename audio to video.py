from IPython.display import HTML
from base64 import b64encode
import os
import shutil
import subprocess

def play(f):
    mp4 = open(f, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)

#Install required packages
subprocess.run(['apt', 'install', 'imagemagick'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['cat', '/etc/ImageMagick-6/policy.xml', '|', 'sed', 's/none/read,write/g', '>', '/etc/ImageMagick-6/policy.xml'])
subprocess.run(['wget', '-qO-', 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x6888550b2fc77d09', '|', 'sudo', 'tee', '/etc/apt/trusted.gpg.d/songrec.asc'])
subprocess.run(['pip', 'install', 'langchain', 'langchain_openai', 'openai', 'requests', 'moviepy==2.0.0.dev2', 'imageio==2.25.1', 'pysrt==1.1.2', 'Pillow==9.5.0', 'ffmpeg-python', 'pytube', 'google-api-python-client', 'google-auth-oauthlib', 'google-auth-httplib2', 'oauth2client', 'git+https://github.com/m1guelpf/auto-subtitle.git'])


#-----------------------------------------------------------------------------------

import os
import json
import pysrt
import string
import random
import requests
from pytube import Search
from base64 import b64encode
from collections import Counter
from IPython.display import HTML
from langchain_openai import ChatOpenAI
from moviepy.video.fx.all import resize
from moviepy.editor import AudioFileClip
from moviepy.editor import AudioFileClip
from moviepy.audio.fx.all import volumex
from langchain_core.prompts import ChatPromptTemplate
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from langchain_core.output_parsers import StrOutputParser
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, CompositeAudioClip, concatenate_videoclips


def play(f):
    mp4 = open(f, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)


#Give local folder path (with multiple audios/videos), local single media file or youtube link
INPUT = "/content/drive/MyDrive/Tips/3 Mistakes Everyone Should Avoid!  (Part -2) ❌ #shorts_Shakeel_p0_i0.5_fr3_rms0.25_pro0.33_rmvpe_mixed/3 Mistakes Everyone Should Avoid!  (Part -2) ❌ #shorts_Shakeel_p0_i0.5_fr3_rms0.25_pro0.33_rmvpe_mixed.wav"  # @param {type:"string"}
#Select the language of input media or leave it as auto for auto detection
Language = "auto"  # @param ["auto", "English", "Urdu", "German"]
#select the desire video format by default = Landscape
FORMAT = "Landscape (Youtube 16:9)"  # @param ["Landscape (Youtube 16:9)", "Portrait  (Tiktok 9:16)", "Square (Instagram 10:10)"]
#This is optional either put direct download link of background music or leave empty
MUSIC = "" # @param {type:"string"}
#This is optional you can leave empty if you do not want to use any Watermark to the video
WATERMARK = "Enterprisium"  # @param {type:"string"}
# Select the Subtitles style to be used in the video or Leave it as it is to use Default (Simple)
SUBTITLES = "Word_Highlight"  # @param ["Word_Highlight", "Word-By-Word", "Simple", "None"]

-------------------------------------------------------INPUT_PROCESSING-------------------------------------------------------------------


# Check if INPUT is a folder path
if os.path.isdir(INPUT):
    audio_files = []
    for root, dirs, files in os.walk(INPUT):
        for file in files:
            if file.endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    for audio_file in audio_files:
        input_basename = os.path.basename(audio_file).split(".")[0]
        output_folder = os.path.join(INPUT_MEDIA, input_basename)
        os.makedirs(output_folder, exist_ok=True)

        # Move the input media to the output subfolder and rename it as voice.mp3
        voice_path = os.path.join(output_folder, "voice.mp3")
        shutil.move(audio_file, voice_path)

        # Get the duration of input media
        audio = AudioFileClip(voice_path)
        duration_in_seconds = round(audio.duration)

        # Print the name of the voiceover file and its duration
        print(f"Processing: {os.path.basename(voice_path)} | Duration: {duration_in_seconds}s")


-------------------------------------------------------CAPTIONING_TRANCRIBING-------------------------------------------------------------------


        !auto_subtitle "{voice_path}" --srt_only "True" --output_dir "{output_folder}" --language "auto"


-----------------------------------------------------SUBTITLE_STYLING-------------------------------------------------------------------

              #Default
        subs = pysrt.open(os.path.join(output_folder, 'voice.srt'))
        for sub in subs:
            sub.text = sub.text.translate(str.maketrans("", "", string.punctuation))
            if len(sub.text) > 20:
                words = sub.text.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) <= 30:
                        current_line += word.title() + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word.title() + " "
                if current_line:
                    lines.append(current_line.strip())
                sub.text = "\n".join(lines)
            else:
                sub.text = sub.text.title()
        subs.save(os.path.join(output_folder, "voice.srt"))

-----------------------------------------------Word_Highlight(Descript like)-------------------------------------------------------------------
#
#
#
#
#
#
#
#
-------------------------------------------------------KEYWORDING-------------------------------------------------------------------


        template = """Imagine you're a highly imaginative artist with the unique ability to map the subjects
        in a given SRT caption to a one word real-world objects and scenes.
        It's important to keep the titles exactly one word, and title must be a real word object or scences
        That human can vision, avoid providing teoritocal titles like: Paradox, Chrysalis, Journey, Reality, Accomplishment
        Instead use real world titles that can bee seen by human eyes like Mansion, Yoga, Car, Money.
        Give me exactly {num_clips} distinct clip titles.
        Each title should seamlessly flow into the next, creating a captivating narrative,
        and each title will be precisely 5 seconds long.
        I want you to understand and imagine the big picture of the video and give me titles that matches
        The entire video, not just invidual scences along Genrate optmised Title, Discription and tags for youtube. The Title For the Video Must be in 80 chracters. Dicriptions must be Very Explanatory discribe everything about the video. discriptions Must not exceed 4500 chracters limit.

        Get inspired by the SRT caption provided:

        {srt_caption}

        Output Instruction:
        Provide only the titles. Each title must be separated by a new line,
        do not mention numbers in titles and titles must be URL-encoded friendly.
        Example Output:
        Yoga
        Forest
        Office
        Jogging
        Sunset
        Cafe
        Hiking
        Spa
        Beach
        Tea
        Luxury
        Money
        """

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo")
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        num_clips = (duration_in_seconds // 5) + 1
        with open(os.path.join(output_folder, "voice.srt"), "r") as f:
            srt_caption = f.read()
            output = chain.invoke({"num_clips": num_clips, "srt_caption": srt_caption})


        clips_titles = output.strip().split("\n")
        clips_titles
-------------------------------------------------------PEXELS_FATCHING-------------------------------------------------------------------
        clip_counter = Counter(clips_titles)
        clips_paths = []
        selected_videos = set()

        for title, count in clip_counter.items():
            headers = {
                "Authorization": os.environ["PEXELS_API_KEY"],
            }
            url = f"https://api.pexels.com/videos/search?query={title}&per_page=15&orientation=portrait&size=medium"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                videos = response.json()["videos"]
                for i in range(count):
                    available_videos = [
                        video
                        for video in videos
                        if video["video_files"][0]["link"] not in selected_videos
                    ]
                    if not available_videos:
                        print(f"No more available videos for title '{title}'.")
                        break

                    video = random.choice(available_videos)
                    video_url = video["video_files"][0]["link"]
                    temp_name = f"{title}_{i}.mp4"
                    video_path = os.path.join(output_folder, temp_name)
                    with open(video_path, "wb") as video_file:
                        video_file.write(requests.get(video_url).content)
                    clips_paths.append(video_path)
                    selected_videos.add(video_url)
            else:
                print(f"Failed to fetch videos for title '{title}'. Status code: {response.status_code}")

        for i in range(num_clips - len(clips_paths)):

            clips_paths.append(clips_paths[0])
-------------------------------------------------------ALINGMENT-------------------------------------------------------------------
        def resize_clip(input_video_path, duration=5, new_dimensions=(720, 1280)):
            video_clip = VideoFileClip(input_video_path)
            total_duration = video_clip.duration
            start_time = (total_duration - duration) / 2
            end_time = start_time + duration
            video_clip = video_clip.subclip(start_time, end_time)

            video_clip = resize(video_clip, new_dimensions)
            return video_clip

        clips = [resize_clip(cp) for cp in clips_paths]


-------------------------------------------------------COMBINATION-------------------------------------------------------------------


        # Initialize an empty music_audio variable
        music_audio = None

        # Check if BG_MUSIC is provided
        if BG_MUSIC.strip():
            # Download the background music
            music_response = requests.get(BG_MUSIC)

            # Save the music file to 'output/music.mp3'
            with open(os.path.join(output_folder, 'music.mp3'), 'wb') as music_file:
                music_file.write(music_response.content)

            # Load and process the background music
            music_audio = AudioFileClip(os.path.join(output_folder, "music.mp3")).fx(volumex, 0.3)  # Adjusted volume level to 0.3
            t_start = int(music_audio.duration // 2 - duration_in_seconds // 2)
            music_audio = music_audio.subclip(t_start, duration_in_seconds + t_start)

        # Load and process the voice audio
        voice_audio = AudioFileClip(os.path.join(output_folder, "voice.mp3")).fx(volumex, 3)

        # Combine the voice audio with background music if it exists
        if music_audio:
            audio = CompositeAudioClip([voice_audio, music_audio])
        else:
            audio = voice_audio

        # Concatenate the video clips
        final_clip = concatenate_videoclips(clips, method="compose").subclip(0, duration_in_seconds)

        # Set the audio for the final clip
        final_clip = final_clip.set_audio(audio)

        # Write the final video file
        final_clip.write_videofile(os.path.join(output_folder, f"{input_basename}.mp4"), codec="libx264", audio_codec="aac", fps=24)
-------------------------------------------------------FINALISATION-------------------------------------------------------------------

        video = VideoFileClip(os.path.join(output_folder, f"{input_basename}.mp4"))
        watermark_clip = TextClip("@Enterprisium", font="Nimbus-Sans-Bold", fontsize=24,
                                  color='white', size=(640, 480)).set_duration(video.duration)
        generator = lambda txt: TextClip(txt, font="Nimbus-Sans-Bold", fontsize=36,
                                         color='white', size=(video.w, video.h),
                                         stroke_color='black', stroke_width=1.5)
        subtitles = SubtitlesClip(os.path.join(output_folder, "voice.srt"), generator)
        result = CompositeVideoClip([video, subtitles.set_position(('center')),
                                     watermark_clip.set_position(('center', 'bottom'))])
        result.write_videofile(os.path.join(output_folder, f"{input_basename}_final.mp4"), fps=video.fps, codec="libx264",
                               audio_codec="aac")

        print(f"Final video saved to: {os.path.join(output_folder, f'{input_basename}_final.mp4')}")

        # Remove temporary files
        for file_path in clips_paths:
            os.remove(file_path)
        if BG_MUSIC.strip():
            os.remove(os.path.join(output_folder, 'music.mp3'))
        os.remove(os.path.join(output_folder, "voice.mp3"))
        os.remove(os.path.join(output_folder, f"{input_basename}.mp4"))
#------------------------END-----------------#
