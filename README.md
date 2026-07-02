# Multimodal Gift Recommender

A gift recommendation chatbot that accepts both **text descriptions** and **camera input** to suggest personalized gifts. Combines semantic search (RAG) with vision understanding (CLIP) and LLM generation.

## How it works

```
User input (text or image)
        ↓
   [CLIP] image → label        [SentenceTransformer] text → vector
        ↓                               ↓
        └──────────→ cosine similarity search over gift catalog
                              ↓
                        top-3 gifts retrieved
                              ↓
                    [GLM-4.5-Air via HuggingFace]
                              ↓
                    personalized recommendation
```

## Features

- **Text mode** — describe the person you're shopping for in natural language
- **Camera mode** — point your camera at an object; CLIP detects it and finds similar gifts
- Kazakh cultural context: CLIP labels include traditional items (piala, dombra, tus kiiz)
- Built with RAG architecture: retrieval → augmentation → generation

## Tech stack

| Component | Technology |
|-----------|-----------|
| Semantic search | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vision understanding | `CLIP` (openai/clip-vit-base-patch32) |
| LLM generation | GLM-4.5-Air via HuggingFace Inference API |
| UI | Streamlit |
| Data | Custom gift catalog (`gifts.csv`) |

## Setup

```bash
git clone https://github.com/melomilk/gift-recommender.git
cd gift-recommender
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
HF_TOKEN=your_huggingface_token_here
```

Run the app:
```bash
# Basic version (text only)
streamlit run recommender.py

# Full version (text + camera)
streamlit run recommender_vision.py
```

## Project structure

```
gift-recommender/
├── recommender.py          # text-based RAG chatbot
├── recommender_vision.py   # multimodal version (text + CLIP camera)
├── data/
│   └── gifts.csv           # gift catalog with descriptions and price ranges
├── requirements.txt
└── .env                    # HF_TOKEN (not committed)
```
