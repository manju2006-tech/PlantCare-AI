
import os
import time
import mimetypes
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("Gemini API key is missing.")

client = genai.Client(api_key=api_key)

image_path = input("Enter plant image path: ").strip()

if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

mime_type, _ = mimetypes.guess_type(image_path)

if not mime_type:
    raise ValueError("Unsupported image format.")

with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()
    
start_time = time.time()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                {
                    "text": """
                    Identify the plant in this image.

                    Return:
                    1. Common plant name
                    2. Scientific name, if identifiable
                    3. Visual evidence supporting identification
                    4. Alternative possible plants
                    5. Identification confidence: High, Medium, or Low

                    If the image does not contain enough information,
                    clearly say that identification is uncertain.
                    Do not invent a plant name.
                    """
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_bytes
                    }
                }
            ]
        }
    ]
)

elapsed_time = time.time() - start_time
print(f"\nResponse time: {elapsed_time:.2f} seconds")

print("\n========== GEMINI PLANT IDENTIFICATION ==========")
print(response.text)
print("=================================================")
