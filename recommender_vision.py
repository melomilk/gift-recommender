from sentence_transformers import SentenceTransformer, util #RAG retrieval
import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient #for my hf token
from dotenv import load_dotenv #api
import os 
from PIL import Image #cv
from transformers import CLIPProcessor, CLIPModel #cv
import torch #deeplearning

load_dotenv()

#loading all models.
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data(_sentence_model):
    df = pd.read_csv('data/gifts.csv')
    gift_vectors = _sentence_model.encode(df['description'].tolist())
    return df, gift_vectors

@st.cache_resource
def load_llm():
    return InferenceClient(api_key=os.getenv("HF_TOKEN"))

@st.cache_resource
def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

sentence_model = load_sentence_model()
df, gift_vectors = load_data(sentence_model)
client = load_llm()
clip_model, clip_processor = load_clip()

#kazakh gift labels for CLIP.

KAZAKH_LABELS = [
    "a piala, traditional Kazakh ceramic tea bowl",
    "a dombra, traditional Kazakh string instrument",
    "a tus kiiz, Kazakh embroidered wall hanging",
    "a keseshka, Kazakh decorative tea cup set",
    "a platok, Kazakh embroidered shawl",
    "a traditional Kazakh jewelry or ornament",
    "a tech gadget or electronic device like headphones or phone",
    "a book or journal",
    "outdoor hiking or fitness equipment",
    "art supplies like paint or sketchbook",
    "a generic household item",
]

#image -> search query.

def image_to_query(pil_image, confidence_threshold=0.25):
    inputs = clip_processor(
        text=KAZAKH_LABELS,
        images=pil_image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=0)

    top_score = probs.max().item()
    top_label = KAZAKH_LABELS[probs.argmax()]

    if top_score < confidence_threshold:
        return None, top_score

    return top_label, top_score

# retrieval + generation

def get_recommendations(query_text):
    user_vector = sentence_model.encode(query_text)
    scores = util.cos_sim(user_vector, gift_vectors)[0]
    df['score'] = scores.numpy()
    top3 = df.nlargest(3, 'score')

    context = ""
    for _, row in top3.iterrows():
        context += f"- {row['name']}: {row['description']}, {row['price_range']}\n"

    prompt = f"""You are a specific, thoughtful gift advisor. Your job is to recommend gifts ONLY from the list provided below — do not suggest anything outside this list.

The user is shopping for someone described as: "{query_text}"

Here are the ONLY gifts you can recommend (choose the best 3):
{context}

For each gift you recommend:
- State the exact gift name from the list
- Explain in 2-3 sentences specifically why it matches this person's personality and interests
- Be concrete, not generic

Do not invent gifts. Only use what's in the list above."""

    response = client.chat.completions.create(
        model="zai-org/GLM-4.5-Air",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048
    )

    return response.choices[0].message.content

# streamlit ui part

st.title("ur personal gift recommender")
st.caption("describe someone or show an object on camera")

# tabs for two input modes
tab1, tab2 = st.tabs(["💬 text search", "📷 camera search"])

#text input goes here
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Describe who you're shopping for..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Finding gifts..."):
            reply = get_recommendations(user_input)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# or camera input goes here
with tab2:
    st.write("Take a photo of an object and we'll find similar gifts.")

    camera_image = st.camera_input("Point camera at an object")

    if camera_image is not None:
        pil_image = Image.open(camera_image)

        with st.spinner("Analyzing image..."):
            detected_label, confidence = image_to_query(pil_image)

        if detected_label is None:
            st.warning(f"Couldn't identify the object clearly (confidence: {confidence:.0%}). Try better lighting or a closer shot.")
        else:
            st.success(f"Detected: **{detected_label}** ({confidence:.0%} confidence)")
            st.write(f"Searching for gifts related to: *{detected_label}*")

            with st.spinner("Finding gifts..."):
                reply = get_recommendations(detected_label)

            st.markdown(reply)