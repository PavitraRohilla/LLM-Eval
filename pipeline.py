
import os
import time
import google.generativeai as genai

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Answer questions accurately and briefly.",
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def query_llm(question: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\n{question}"
    start = time.perf_counter()
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    latency_ms = (time.perf_counter() - start) * 1000
    prompt_tokens     = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    return {
        "answer":            response.text.strip(),
        "model":             model,
        "latency_ms":        round(latency_ms, 2),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd":          0.0,
    }
