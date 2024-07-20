import captacity
from moviepy.config import change_settings

# Set the path to the ImageMagick binary (adjust if necessary)
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"}) 

captacity.add_captions(
    video_file="/content/1.mp4",
    output_file="/content/short_with_captions.mp4",
    font = "/content/Ubuntu-Bold.ttf",
    font_size = 50,
    font_color = "yellow",
    stroke_width = 3,
    stroke_color = "black",
    shadow_strength = 1.0,
    shadow_blur = 0.1,
    highlight_current_word = True,
    word_highlight_color = "red",
    line_count=1,
    padding = 50,
    use_local_whisper=True
)
