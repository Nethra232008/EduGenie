import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Create FastAPI app
app = FastAPI(title="EduGenie")

# HTML templates
templates = Jinja2Templates(directory="templates")

# Gemini model
MODEL = "gemini-3.5-flash-lite"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


@app.post("/qa")
async def question_answer(question: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        You are EduGenie, an educational AI assistant.

        Answer the student's question clearly and accurately.
        Use simple language suitable for a college student.

        Student question:
        {question}
        """
    )

    return {"answer": response.text}


@app.post("/explain")
async def explain_concept(concept: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        Explain the following concept like a friendly teacher.

        Concept:
        {concept}

        Include:
        1. Simple definition
        2. Easy explanation
        3. Real-world example
        4. Short summary
        """
    )

    return {"explanation": response.text}


@app.post("/summarize")
async def summarize(text: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        Summarize the following study material.

        Make the summary:
        - Clear
        - Short
        - Easy to understand
        - Useful for revision

        Study material:
        {text}
        """
    )

    return {"summary": response.text}


@app.post("/quiz")
async def generate_quiz(topic: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        Create a quiz for a college student about:

        {topic}

        Create exactly 3 multiple-choice questions.

        Each question must have:
        - The question
        - 4 options
        - The correct answer
        - A short explanation

        Format the result clearly.
        """
    )

    return {"quiz": response.text}


@app.post("/learn/recommendations")
async def learning_path(topic: str):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
        Create a personalized learning path for a student who wants
        to learn:

        {topic}

        Organize it from:
        Beginner → Intermediate → Advanced

        For each level include:
        - Topics to learn
        - What the student should understand
        - Suggested practice activities
        """
    )

    return {"learning_path": response.text}