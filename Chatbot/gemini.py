"""
Integrate different large language models as global accessible utilities
(Support expert hierarchy using Gemini 1.5 Pro and Flash integration)

Criteria:
    + Successfully integrate Gemini 1.5 Pro and Gemini 1.5 Flash API
    + Implement error handling and rate limiting
    + Create a standardized interface for accessing Gemini 1.5 Pro's features
    + Create a standardized interface for accessing Gemini 1.5 Flash's festures
    + Benchmark response times to ensure they meet the expected 0.7-second average

Research:
    + https://ai.google.dev/gemini-api/docs/quickstart?lang=python
    + https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_pro.ipynb

"""

import google.generativeai as genai
import os

class GeminiPro:
    _instance = None

    # Ensure only one instance of the class exists (Singleton Pattern)
    def __new__(cls):                                          # control creation of new instances of class        
        if cls._instance is None:                              # ensure class has not yet been instantiated
            cls._instance = super(GeminiPro, cls).__new__(cls) # create new instance of class
            cls._instance.__init__()                           # initialize instance
        return cls._instance                                   # Return single instance of class

    def __init__(self):
        # Load API key
        api_key = os.environ.get("GEMINI_API_KEY")

        # Load Gemini-1.5-Pro model
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
    def query(self, prompt, max_tokens=150):
        response = self.model.generate_content(prompt)
        return response.candidates[0].text.strip()
        
class GeminiFlash:
    _instance = None 

    # Same pattern as GeminiPro
    def __new__(cls):                                          
        if cls._instance is None:                             
            cls._instance = super(GeminiPro, cls).__new__(cls) 
            cls._instance.__init__()                          
        return cls._instance                                   

    def __init__(self):
        # Load API key
        api_key = os.environ.get("GEMINI_API_KEY")

        # Load Gemini-1.5-Pro model
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def query(self, prompt, max_tokens=150):
        response = self.model.generate_content(prompt)
        return response.candidates[0].text.strip()
    
def main():
    prompt = "What is the difference between a stack and a queue?"
    q1 = GeminiPro.query(prompt)
    q2 = GeminiFlash.query(prompt)
    print(f"Prompt: {prompt}\nGemini Pro: {q1}\nGemini Flash: {q2}")

main()

        
