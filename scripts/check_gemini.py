# resolveiq\scripts\check_gemini.py
"""Small probe: can this key generate, can it embed, and how long does each take?"""
import time

from google import genai

from resolveiq.config import get_settings

settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

started = time.perf_counter()
answer = client.models.generate_content(
    model=settings.gemini_model,
    contents="Reply with exactly three words that describe what a triage engine does.",
)
print(f"generate  {settings.gemini_model}  {time.perf_counter() - started:.2f}s")
print("  ->", (answer.text or "").strip()[:200])

started = time.perf_counter()
vector = client.models.embed_content(
    model=settings.gemini_embed_model,
    contents="no hot water in the apartment since monday",
).embeddings[0].values
print(f"embed     {settings.gemini_embed_model}  {time.perf_counter() - started:.2f}s")
print(f"  dims={len(vector)}  first3={[round(v, 4) for v in vector[:3]]}")
match = "MATCH" if len(vector) == settings.embedding_dimensions else "FIX .env"
print(f"  .env says {settings.embedding_dimensions} -> {match}")