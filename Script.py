import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyC6N1MVe9WmAFjWMNuXjlaLnYa8eO"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def gpt_entry(niche):
    prompt = f"Generate 5 short video ideas about {niche}. Each idea should be a single sentence without numbering."
    response = model.generate_content(prompt)
    ideas = response.text.split('\n')
    return [idea.strip() for idea in ideas if idea.strip() and not idea.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]

def gpt_reformat(idea):
    prompt = f"Create a short script for a 60-second video based on this idea: {idea}. The script should have an introduction, 3 main points, and a conclusion."
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    test_niche = "artificial intelligence"
    ideas = gpt_entry(test_niche)
    print("Generated Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea}")
    
    if ideas:
        print("\nGenerating script for the first idea:")
        script = gpt_reformat(ideas[0])
        print(script)
