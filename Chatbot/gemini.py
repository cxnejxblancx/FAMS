"""
Integrate different large language models as global accessible utilities
(Support expert hierarchy using Gemini 1.5 Pro and Flash integration)

1. Initialize and activate virtual environment
2. Install required packages: pip install -r requirements.txt
3. Assign API key as environment variable: export API_KEY="AIzaSyCrGtK_-qzVe2qhvSNNaYyyx82YlamldtY" ; manual setup for Windows


Criteria:
    + Successfully integrate Gemini 1.5 Pro and Gemini 1.5 Flash API
    + Implement error handling and rate limiting
    + Create a standardized interface for accessing Gemini 1.5 Pro's features
    + Create a standardized interface for accessing Gemini 1.5 Flash's festures
    + Benchmark response times to ensure they meet the expected 0.7-second average

Research:
    + https://ai.google.dev/gemini-api/docs/quickstart?lang=python
    + https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_pro.ipynb
    + https://ai.google.dev/pricing#1_5pro

Updates:
    + Gemini Pro and Flash models successfully integrated
    + Error handling and rate limiting implemented
    + Standardized interfaces created for both models
    + Next: Benchmark response times
"""

import google.generativeai as genai
import os
import logging
from ratelimit import limits, sleep_and_retry

# Setup logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s, %(levelname)s: %(message)s")

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
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.error("GEMINI_API_KEY is missing from envrionment variables.")
            raise ValueError("GEMINI_API_KEY is undefined.")

        # Load Gemini-1.5-Pro model
        try:
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        except Exception as e:
            logging.error(f"Failed to load Gemini-1.5-Pro model: {e}")
            raise RuntimeError("An error occured while loading Gemini-1.5-Pro model.")
        
    # Set rate limit to 2 RPM
    @sleep_and_retry
    @limits(calls=2, period=60)
    def query(self, prompt, max_tokens=150):
        try:
            response = self.model.generate_content(prompt)
            return response.candidates[0].text.strip()
        except Exception as e:
            logging.error(f"Failed to query Gemini-1.5-Pro model: {e}")
            return f"Failed to query Gemini-1.5-Flash model: {e}"
        
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
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.error("GEMINI_API_KEY is missing from envrionment variables.")
            raise ValueError("GEMINI_API_KEY is undefined")

        # Load Gemini-1.5-Pro model
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.error(f"Failed to load Gemini-1.5-Flash model: {e}")
            raise RuntimeError(f"Failed to load Gemini-1.5-Flash model: {e}")
    
    # Set rate limit to 15 RPM
    @sleep_and_retry
    @limits(calls=15, period=60) #15 RPM
    def query(self, prompt, max_tokens=150):
        try:
            response = self.model.generate_content(prompt)
            return response.candidates[0].text.strip()
        except Exception as e:
            logging.error(f"Failed to query Gemini-1.5-Flash model: {e}")
            return f"Failed to query Gemini-1.5-Flash model: {e}"
    
def main():
    prompt = "What is the difference between a stack and a queue?"
    q1 = GeminiPro.query(prompt)
    q2 = GeminiFlash.query(prompt)
    print(f"Prompt: {prompt}\nGemini Pro: {q1}\nGemini Flash: {q2}")
main()

        
