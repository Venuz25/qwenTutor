import google.generativeai as genai

genai.configure(api_key="AIzaSyDGgitNOwLpy3fb5GBjWLEVoDD5W8ATVZ0")

print("Modelos disponibles para generar texto:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)