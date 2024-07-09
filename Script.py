import os
from googleapiclient.discovery import build

google_api_key = os.getenv('GOOGLE_API_KEY')
service = build('gemini', 'v1', developerKey=google_api_key)

def gpt_entry(niche):
    ideas = []
    request = service.text().generate(
        body={
            'prompt': f"You are a YouTube content creator. You are making videos about {niche}. I need 5 title ideas for my YouTube channel about {niche}. Format the titles into 5 bullet points."
        }
    )
    response = request.execute()
    results = response['choices'][0]['text'].splitlines()

    for result in results:
        ideas.append(result)
    return ideas

def gpt_script(subject):
    request = service.text().generate(
        body={
            'prompt': f"You are creating voice-overs for short videos(about 1 minute long), you sound very clever and also funny. I need short voice-over text for about 1500 characters about: {subject}. I don't need any instructions according to video and anything else."
        }
    )
    response = request.execute()
    script = response['choices'][0]['text']
    return script

def gpt_reformat(subject):
    script = gpt_script(subject)
    request = service.text().generate(
        body={
            'prompt': f"{script} reformat this script into one paragraph remove any instructions to the video, i need just pure text."
        }
    )
    response = request.execute()
    script = response['choices'][0]['text']
    return script
