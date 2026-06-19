import streamlit as st
import pandas as pd
from google import genai

st.set_page_config(
    page_title="GenAI Search Insights",
    layout="wide"
)

st.title("GenAI Search Insights")

client = genai.Client(
    api_key=st.secrets["api_key"]
)

df = pd.read_csv("datasets/search_queries.csv")

top_queries = (
    df.sort_values("Conversions", ascending=False)
    .head(20)
)

query_text = "\n".join(
    top_queries["Query"].tolist()
)

prompt = f"""
You are a Paid Search Expert.

Analyze these search queries and provide:

1. Customer Intent
2. High Intent Queries
3. Low Intent Queries
4. Budget Recommendations
5. Bid Optimization Suggestions
6. Search Trends

Queries:

{query_text}
"""

# ==========================================
# SESSION STATE CACHE
# Prevents re-calling Gemini on every
# Streamlit re-render / widget interaction
# ==========================================

if "genai_insights_result" not in st.session_state:
    st.session_state.genai_insights_result = None

if st.button("Generate AI Insights"):
    with st.spinner("Generating AI Insights..."):

        # ---- FIX: single try/except block ----
        # Previously response.text was written
        # twice — once inside try and again after
        # the block. If the API call failed,
        # response was undefined and threw NameError.

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            # Store in session state so result
            # persists across re-renders
            st.session_state.genai_insights_result = response.text
            st.success("AI Insights Generated")

        except Exception as e:
            st.error(f"AI Service Temporarily Unavailable: {e}")
            st.session_state.genai_insights_result = None

# ==========================================
# DISPLAY — reads from session state,
# not from a live API call
# ==========================================

if st.session_state.genai_insights_result:

    st.subheader("🤖 GenAI Search Insights")

    st.markdown(
        st.session_state.genai_insights_result
    )

else:

    st.info(
        "Click **Generate AI Insights** above to run the analysis."
    )

