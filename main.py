from voice_gen import generate_voice_over
from Script import gpt_entry, gpt_reformat
from video_make import make_vid
from add_subs import add_subtitles

if __name__ == "__main__":
    niche = input("Choose your niche (e.g., 'making money'): ")
    ideas = gpt_entry(niche)
    
    print("Generated Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea}")
    
    while True:
        choice = input(f'Pick your desired video (from 1 to {len(ideas)}): ')
        try:
            choice = int(choice)
            if 1 <= choice <= len(ideas):
                break
            else:
                print(f'Wrong input, type in a value between 1 and {len(ideas)}')
        except ValueError:
            print("Please enter a valid number.")
    
    script = gpt_reformat(ideas[choice - 1])
    print("\nGenerated Script:")
    print(script)
    
    generate_voice_over(script)
    make_vid(niche)
    add_subtitles('voice_over.mp3')
