"""
Integrate different large language models as global accessible utilities
(Support expert hierarchy using Gemini 1.5 Pro and Flash integration)

1. Initialize virtual environment: python3 -m venv myvenv
2. Activate virtual environment: source ./myvenv/bin/activate
3. Install required packages: pip install -r requirements.txt
4. Assign API key as environment variable

Criteria:
    + Successfully integrate Gemini 1.5 Pro and Gemini 1.5 Flash API
    + Implement error handling and rate limiting
    + Create a standardized interface for accessing Gemini 1.5 Pro's features
    + Create a standardized interface for accessing Gemini 1.5 Flash's festures
    + Benchmark response times to ensure they meet the expected 0.7-second average

Research:
    + https://refactoring.guru/design-patterns/singleton/python/example#:~:text=Singleton%20is%20a%20creational%20design,the%20modularity%20of%20your%20code.
    + https://www.geeksforgeeks.org/singleton-pattern-in-python-a-complete-guide/
    + https://ai.google.dev/gemini-api/docs/quickstart?lang=python
    + https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_1_5_pro.ipynb
    + https://ai.google.dev/pricing#1_5pro
    + https://youtu.be/FoxTh4KSDJ8
    + https://pynative.com/python-get-execution-time-of-program/

Updates:
    + Gemini Pro and Flash models successfully integrated
    + Error handling and rate limiting implemented
    + Standardized interfaces created for both models
    + Next: Benchmark response times, currently has 0.9s average
"""

import google.generativeai as genai
import os
import logging
from ratelimit import limits, sleep_and_retry
import time

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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY is missing from envrionment variables.")
            raise ValueError("GEMINI_API_KEY is undefined.")
        genai.configure(api_key=api_key)

        # Load Gemini-1.5-Pro model
        try:
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        except Exception as e:
            logging.error(f"Failed to load Gemini-1.5-Pro model: {e}")
            raise RuntimeError(f"Failed to load Gemini-1.5-Pro model: {e}")
   
    # Set rate limit to 2 RPM
    @sleep_and_retry
    @limits(calls=2, period=60)
    def query(self, prompt):
        try:
            start = time.time()
            response = self.model.generate_content(prompt)
            end = time.time()
            elapsed_time = end - start
            print(f"Gemini-1.5-Pro Response Time: {elapsed_time}s")
            return response.text.strip(), elapsed_time
        except Exception as e:
            logging.error(f"Failed to query Gemini-1.5-Pro model: {e}")
            return f"Failed to query Gemini-1.5-Pro model: {e}"
        


class GeminiFlash:
    _instance = None 
    
    # Same pattern as GeminiPro
    def __new__(cls):                                          
        if cls._instance is None:                             
            cls._instance = super(GeminiFlash, cls).__new__(cls) 
            cls._instance.__init__()                          
        return cls._instance                                   

    def __init__(self):
        # Load API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY is missing from envrionment variables.")
            raise ValueError("GEMINI_API_KEY is undefined")
        genai.configure(api_key=api_key)
        # Load Gemini-1.5-Pro model
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.error(f"Failed to load Gemini-1.5-Flash model: {e}")
            raise RuntimeError(f"Failed to load Gemini-1.5-Flash model: {e}")
    
    # Set rate limit to 15 RPM
    @sleep_and_retry
    @limits(calls=15, period=60) #15 RPM
    def query(self, prompt):
        try:
            start = time.time()
            response = self.model.generate_content(prompt)
            end = time.time()
            elapsed_time = end - start
            print(f"Gemini-1.5-Flash Response Time: {elapsed_time}s")
            return response.text.strip(), elapsed_time
        except Exception as e:
            logging.error(f"Failed to query Gemini-1.5-Flash model: {e}")
            return f"Failed to query Gemini-1.5-Flash model: {e}"

def main():
    prompt = "What is the difference between a stack and a queue? Answer in 1 sentence."
    q1 = GeminiPro().query(prompt)
    q2 = GeminiFlash().query(prompt)
    print(f"Prompt: {prompt}\nGemini Pro: {q1[0]}\nGemini Flash: {q2[0]}\nAverage Response Time: {(q1[1]+q2[1]) / 2}s")
main()
