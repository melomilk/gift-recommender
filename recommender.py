from sentence_transformers import SentenceTransformer, util
import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

#loading all models.

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    model = load_model()
    df = pd.read_csv('data/gifts.csv')
    gift_vectors = model.encode(df['description'].tolist())
    return df, gift_vectors

@st.cache_resource
def load_llm():
    return InferenceClient(
        api_key=os.getenv("HF_TOKEN")
    )

model = load_model()
df, gift_vectors = load_data()
client = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("describe who you're shopping for..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    #encode the query. (words -> vectors)
    user_vector = model.encode(user_input) #384 numbers

    #r - retrieve
    #basically cosine similarity measures how much vectors point in the same direction
    scores = util.cos_sim(user_vector, gift_vectors)[0]
    df['score'] = scores.numpy()
    print(df[['name', 'description', 'score']].sort_values('score', ascending=False).head(10))
    top3 = df.nlargest(3, 'score')

    #a - augment
    #essentially formats the 3 retrieved gifts into a readable string for a user
    context = ""
    for _, row in top3.iterrows():
        context += f"- {row['name']}: {row['description']}, {row['price_range']}\n"

    # build prompt
    prompt = f"""You are a smart, thoughtful gift advisor. Your job is to recommend gifts ONLY from the list provided below — do not suggest anything outside this list.

The user is shopping for someone described as: "{user_input}"

Here are the ONLY gifts you can recommend (choose the best 3):
{context}

For each gift you recommend:
- State the exact gift name from the list
- Explain in 2-3 sentences specifically why it matches this person's personality and interests
- Be concrete, not generic

Do not invent gifts. Only use what's in the list above."""

    #g - generate 
    #basically writes the response
    response = client.chat.completions.create(
        model="zai-org/GLM-4.5-Air",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048
    )

    reply = response.choices[0].message.content



    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)