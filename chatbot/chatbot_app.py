import mysql.connector
import streamlit as st        
import cohere
from datetime import datetime

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Smart Chatbot", page_icon="🤖", layout="centered")

# --- PAGE HEADER ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🤖 Smart Chatbot Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask me anything. I’m here to help you 24x7! 🌟</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Cohere API Client ---
co = cohere.Client("***")  # Replace with your API Key

# --- USER INPUT BOX ---
user_msg = st.text_input("💬 Your Question:", placeholder="Type your question here...")

# --- FUNCTION TO GENERATE REPLY ---
def get_bot_reply(prompt):
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# --- MAIN LOGIC ---
if user_msg:
    with st.spinner("🤔 Thinking..."):
        bot_reply = get_bot_reply(user_msg)

    # --- Display Result ---
    st.markdown("### 🧠 Bot Says:")
    st.success(bot_reply)

    # --- Save to MySQL ---
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="0mm...123",  # Change this!
            database="chatdb"
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_log (user_msg, bot_reply) VALUES (%s, %s)",
            (user_msg, bot_reply)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
