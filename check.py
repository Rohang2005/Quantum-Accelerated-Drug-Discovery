import google.generativeai as genai

genai.configure(api_key="AIzaSyAI-tCA4WZwfHNpnrAa2F_CeaGaHctYK1E")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
