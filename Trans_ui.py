import streamlit as st
import os
from google import genai
from function import analyze_sentiment,classify_intent,extract_entities,extract_keywords,nlp_pipeline
import json
from io import BytesIO
import pandas as pd
# --- Initialize Gemini Client ---
st.sidebar.header("Configuration")
st.sidebar.write("Ensure your GEMINI_API_KEY environment variable is set.")

try:
    client = genai.Client()
except Exception as e:
    st.error(f"Error initializing client: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"

# --- Streamlit App ---
st.title("Audio Transcription & Summarization with Gemini 2.5 Flash")

st.write("""
Upload an audio file (e.g., WAV, MP3) and get both a verbatim transcription and a concise summary.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])

PROMPT = """**Prompt for AI Transcription and Summarization:** Transcribe the provided audio file accurately, ensuring to capture all spoken words and nuances.Once the transcription is complete, summarize the content in clear, coherent English sentences.The summary should encapsulate the main ideas and key points discussed in the audio while maintaining proper sentence structure and grammatical correctness.The final output should consist solely of the English summary, devoid of any transcription details or audio references."""

if uploaded_file:
    st.info("Uploading and processing the audio... This may take a few moments.")
    
    # --- Step 1: Save uploaded file temporarily ---
    temp_file_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # --- Step 2: Upload audio to Gemini ---
        audio_file = client.files.upload(file=temp_file_path)
        st.success(f"File uploaded successfully: {audio_file.name}")

        # --- Step 3: Generate transcription and summary ---
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[PROMPT, audio_file]
        )

        result_text = response.text
        insights=nlp_pipeline(result_text)
        
        
        df = pd.DataFrame({
        "Key": insights.keys(),
        "Value": [str(v) for v in insights.values()]  # Convert lists/dicts to string
    })
    
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Insights')
        processed_data = output.getvalue()
        
    
    # -----------------------------
    # 4️⃣ Download button
    # -----------------------------
        st.download_button(
        label="📥 Download Insights as Excel",
        data=processed_data,
        file_name="call_insights.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


        

        # --- Step 4: Display results ---
        st.subheader("Transcription & Summary")
        st.text_area("Output:", value=result_text, height=400)

        # --- Step 5: Provide download option ---
        st.download_button(
            label="Download Output as Text",
            data=result_text,
            file_name="transcription_summary.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error processing audio: {e}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass
