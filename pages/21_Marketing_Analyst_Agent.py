import streamlit as st
import pandas as pd
from google import genai

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Marketing Analyst Agent",
    layout="wide"
)

st.title(
    "AI Marketing Analyst Agent"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
"""
Ask questions such as:

• Analyze campaign performance

• Which campaigns should receive more budget?

• Identify inefficient campaigns

• Analyze search query performance

• Summarize A/B test results

• Analyze AI search visibility

• Give executive recommendations
"""
)

# ==========================================
# OPENAI CLIENT
# ==========================================

client = genai.Client(
api_key = st.secrets["api_key"])

# ==========================================
# LOAD DATA
# ==========================================

campaigns = pd.read_csv(
    "datasets/campaigns.csv"
)

keywords = pd.read_csv(
    "datasets/keywords.csv"
)

queries = pd.read_csv(
    "datasets/search_queries.csv"
)

ab_test = pd.read_csv(
    "datasets/ab_test_data.csv"
)

visibility = pd.read_csv(
    "datasets/ai_visibility.csv"
)

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Spend",
    f"${campaigns['Spend'].sum():,.0f}")

col2.metric(
    "Total Revenue",
    f"${campaigns['Revenue'].sum():,.0f}")

col3.metric(
    "Average ROAS",
    round(
        campaigns["ROAS"].mean(),
        2))

col4.metric(
    "Average CPA",
    round(
        campaigns["CPA"].mean(),
        2))

# ==========================================
# DATA SUMMARIES
# ==========================================

campaign_summary = (
    campaigns
    .groupby("Campaign_Type")
    .agg({
        "Spend":"sum",
        "Revenue":"sum",
        "Conversions":"sum",
        "ROAS":"mean",
        "CPA":"mean"
    })
    .reset_index()
)

top_keywords = (
    keywords
    .sort_values(
        "Conversions",
        ascending=False
    )
    .head(20)
)

top_queries = (
    queries
    .sort_values(
        "Conversions",
        ascending=False
    )
    .head(20)
)

visibility_summary = (
    visibility
    .groupby("Brand_Mentioned")
    .size()
    .reset_index(name="Mentions")
)

ab_test_summary = (
    ab_test
    .groupby("Variant")
    .agg(
        Users=("User_ID", "count"),
        Conversions=("Converted", "sum")
    )
    .reset_index()
)

ab_test_summary["Conversion_Rate"] = (
    ab_test_summary["Conversions"]
    /
    ab_test_summary["Users"]
)

# ==========================================
# USER QUESTION
# ==========================================

analysis_type = st.selectbox(
    "Analysis Type",
    [
        "Campaign Performance",
        "Budget Optimization",
        "Keyword Intelligence",
        "Search Query Intelligence",
        "AI Visibility Review",
        "Executive Summary"
    ]
)

question = st.text_area(
    "Ask the AI Analyst",
    height=120
)


# ==========================================
# Opportunity Detection
# ==========================================

high_roas_campaigns = campaigns[
    campaigns["ROAS"]
    >
    campaigns["ROAS"].quantile(0.75)
]

high_cpa_campaigns = campaigns[
    campaigns["CPA"]
    >
    campaigns["CPA"].quantile(0.75)
]

low_roas_campaigns = campaigns[
    campaigns["ROAS"]
    <
    campaigns["ROAS"].quantile(0.25)
]

top_queries_summary = (
    queries
    .sort_values(
        "Conversions",
        ascending=False
    )
    .head(10)
)

# ==========================================
# ANALYZE BUTTON
# ==========================================
if st.button("Generate Analysis"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    with st.spinner("Analyzing marketing performance..."):
        prompt = f"""
        You are a Senior Paid Search Analytics Manager.
        ANALYSIS TYPE:
        {analysis_type}

        USER QUESTION:
        {question}

        CAMPAIGN SUMMARY:
        {campaign_summary.to_string(index=False)}

        TOP KEYWORDS:
        {top_keywords.to_string(index=False)}

        TOP SEARCH QUERIES:
        {top_queries.to_string(index=False)}

        AI SEARCH VISIBILITY:
        {visibility_summary.to_string(index=False)}

        HIGH ROAS CAMPAIGNS:
        {high_roas_campaigns.head(20).to_string(index=False)}

        HIGH CPA CAMPAIGNS:
        {high_cpa_campaigns.head(20).to_string(index=False)}

        LOW ROAS CAMPAIGNS:
        {low_roas_campaigns.head(20).to_string(index=False)}

        TOP CONVERTING SEARCH QUERIES:
        {top_queries_summary.to_string(index=False)}

        AB TEST RESULTS:
        {ab_test_summary.to_string(index=False)}

        Provide:

        1. Executive Summary
        2. Key Findings
        3. Risks
        4. Opportunities
        5. Budget Recommendations
        6. Search Strategy Recommendations
        7. Experimentation Ideas
        8. Estimated Business Impact

        Be specific, quantitative, and business-oriented.
        """

        try:
            response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 4000,
                "top_p": 0.9})
            
            result = response.text

            st.success(
                "Analysis Complete"
            )

            st.subheader(
                "AI Analyst Report"
            )

            st.markdown(result)

            st.session_state.messages.append({
                "question": question,
                "answer": result})
            
            st.session_state.messages = (
                st.session_state.messages[-10:])

        except Exception as e:
            st.error(
                f"Gemini Error: {str(e)}"
            )

        st.subheader("Previous Analyses")

        if len(st.session_state.messages) == 0:
            st.info(
                "No previous analyses yet.")
        else:
            for msg in reversed(st.session_state.messages):
                with st.expander(msg["question"]):
                    st.markdown(
                        msg["answer"]
                    )


