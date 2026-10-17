import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

app = FastAPI(title="CampusPilot GenAI Backend")

# Enable CORS so your frontend HTML can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the GenAI client with your API key
# Recommendation: Set GEMINI_API_KEY as an environment variable or paste directly here for your hackathon prototype
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=API_KEY)

class ChatRequest(BaseModel):
    prompt: str
    language: Optional[str] = "en"
    image: Optional[str] = None  # Base64 data URL from camera capture

@app.post("/api/ai-chat")
async def ai_chat(request: ChatRequest):
    try:
        # Construct system instructions to maintain the persona and grounding
        lang_instruction = f" Respond in {request.language} dialect/language." if request.language and request.language != "en" else ""
        system_prompt = (
            "You are CampusPilot AI, a helpful undergraduate academic assistant for college students. "
            "Provide accurate, clear explanations for coursework, notes, timetables, and notices."
            f"{lang_instruction}"
        )

        contents = []
        
        # Handle camera snapshot image if provided from frontend
        if request.image and "," in request.image:
            header, encoded = request.image.split(",", 1)
            import base64
            image_bytes = base64.b64decode(encoded)
            # Determine mime type from header (e.g. image/png)
            mime_type = header.split(";")[0].replace("data:", "")
            
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )
            contents.append("Analyze this image of a problem, diagram, or equation and explain the solution step-by-step.")

        # Append user prompt text
        if request.prompt:
            contents.append(request.prompt)

        # Call Gemini model (using gemini-2.5-flash for fast multimodal text and vision tasks)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
            ),
        )

        return {
            "status": "success",
            "response": response.text,
            "source": "Gen AI API & Campus Knowledge Base"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)