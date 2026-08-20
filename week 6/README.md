# Evaluator–Generator RAG Platform

A self-evaluating, iterative question-answering platform: a **Generator LLM**
retrieves context and drafts an answer; an **Evaluator LLM** judges it and
either accepts it or sends feedback back for revision — up to **4** loops.

## Architecture -> Code map

| Diagram component            | File(s) |
|-------------------------------|---------|
| Ingestion pipeline (Load/Extract/Chunk/Embed/Store) | `app/ingestion/loaders.py`, `app/ingestion/pipeline.py` |
| LLM External Knowledge (vector store) | `app/vectorstore/store.py` |
| Generator LLM                 | `app/agents/generator.py` |
| Evaluator LLM                 | `app/agents/evaluator.py` |
| Generator Memory (isolated)   | `app/memory/generator_memory.py` |
| Evaluator Memory (isolated)   | `app/memory/evaluator_memory.py` |
| Feedback loop, max 4 iterations | `app/workflow/orchestrator.py` |
| Redis cache (docs/embeddings/responses/evaluations) | `app/cache/redis_cache.py` |
| API + UI                      | `app/main.py`, `frontend/index.html` |

LCEL is used inside each agent: `generator_chain = GENERATOR_PROMPT | llm | StrOutputParser()`
and `evaluator_chain = EVALUATOR_PROMPT | llm | JsonOutputParser()`. The
orchestrator composes these two LCEL chains into the imperative loop
required to enforce the max-loop-count rule (a pure `|` pipeline can't
express "stop after N conditional retries" cleanly, so this part is a
plain Python loop calling the two chains).

## Setup (copy-paste, in order)

By default this runs on **Groq's free tier** for chat (fast, no local compute)
and a **free local embedding model** for the vector store (no signup needed
for that part). No credit card required anywhere.

### 1. Get the code and enter the folder
```bash
cd evaluator_generator_rag
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at https://console.groq.com/keys (no credit card) and copy your key.

### 4. Start Redis
```bash
docker compose up -d
```

### 5. Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and set `GROQ_API_KEY=gsk_...` — everything else can stay default.

### 6. Run the platform
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Open the UI
Visit **http://localhost:8000** — upload a PDF/DOCX/TXT/code file/PPTX/WAV,
or paste a URL, or a Wikipedia topic, then ask a question.

## Switching providers later

Everything reads from `.env` through `app/llm_provider.py` — no code
changes needed to switch:

**Fully offline (Ollama)** — no signup, but slower on CPU:
```
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
```
(requires `ollama pull llama3.2` and `ollama pull nomic-embed-text` first)

**OpenAI (paid, best quality)**:
```
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

## API reference (for testing without the UI)

```bash
# Health check (also pings Redis)
curl http://localhost:8000/health

# Ingest a file
curl -F "file=@/path/to/doc.pdf" http://localhost:8000/ingest/file

# Ingest a URL
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Ingest a Wikipedia page
curl -X POST http://localhost:8000/ingest/wikipedia \
  -H "Content-Type: application/json" \
  -d '{"topic": "Retrieval-augmented generation"}'

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "question": "What is RAG?"}'
```

The `/ask` response includes `final_answer`, `status`
(`accepted` or `max_loops_reached`), `loops_used`, and a `history` array
showing every generator/evaluator round trip — useful for grading/demoing
the iterative loop itself.

## Grounding & unsupported-question handling

The Generator's system prompt explicitly instructs it to say the
information isn't available rather than invent an answer when retrieval
returns nothing relevant (`app/agents/generator.py`). The Evaluator
independently penalizes unsupported claims, so a hallucinated answer
would also fail evaluation and get sent back for revision.

## Memory isolation

`GeneratorMemory` and `EvaluatorMemory` are separate classes with no
shared references, writing to distinct Redis key prefixes
(`gen_memory:*` vs `eval_memory:*`). Neither class exposes a way to read
the other's keys, which is what prevents accidental cross-access.

## Caching strategy (why it won't serve stale answers)

- **Documents**: cached by content hash (`app/ingestion/pipeline.py`) — if
  you re-upload the exact same file, ingestion is skipped; a changed file
  gets a new hash and is re-ingested normally.
- **Embeddings**: cached by exact text (`app/vectorstore/store.py`).
- **LLM responses**: only the *first-pass* generator answer is cached
  (keyed on question + retrieved context); any revision triggered by
  evaluator feedback always calls the LLM fresh, so the feedback loop is
  never short-circuited by a stale cached answer.
- **Evaluations**: cached by question + exact answer text, so identical
  answers aren't re-judged, but any different revision is evaluated fresh.

## Extending it

- Swap `ChatOpenAI` for any other LangChain-supported chat model in
  `app/agents/generator.py` / `evaluator.py`.
- Swap Chroma for another vector DB by editing `app/vectorstore/store.py`
  only — nothing else depends on Chroma directly.
- Tune `MAX_LOOPS`, `RETRIEVAL_K`, `CACHE_TTL` in `.env` without touching code.
