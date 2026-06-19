import streamlit as st
import pandas as pd
import re
from google import genai
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

st.set_page_config(
    page_title="Executive Insights Generator",
    layout="wide"
)

st.title("Executive Insights Generator")

# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=st.secrets["api_key"]
)

# ==========================================
# LOAD DATA
# ==========================================

campaigns  = pd.read_csv("datasets/campaigns.csv")
keywords   = pd.read_csv("datasets/keywords.csv")
queries    = pd.read_csv("datasets/search_queries.csv")
ab_test    = pd.read_csv("datasets/ab_test_data.csv")
visibility = pd.read_csv("datasets/ai_visibility.csv")

# ==========================================
# KPI SECTION
# ==========================================

total_spend   = campaigns["Spend"].sum()
total_revenue = campaigns["Revenue"].sum()
avg_roas      = campaigns["ROAS"].mean()
avg_cpa       = campaigns["CPA"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Spend",    f"${total_spend:,.0f}")
col2.metric("Total Revenue",  f"${total_revenue:,.0f}")
col3.metric("Average ROAS",   round(avg_roas, 2))
col4.metric("Average CPA",    round(avg_cpa, 2))

# ==========================================
# SUMMARIES
# ==========================================

campaign_summary = (
    campaigns
    .groupby("Campaign_Type")
    .agg({
        "Spend": "sum", "Revenue": "sum",
        "Conversions": "sum", "ROAS": "mean", "CPA": "mean"
    })
    .reset_index()
)

top_queries  = queries.sort_values("Conversions", ascending=False).head(20)
top_keywords = keywords.sort_values("Conversions", ascending=False).head(20)

visibility_summary = (
    visibility
    .groupby("Brand_Mentioned")
    .size()
    .reset_index(name="Mentions")
)

ab_summary = (
    ab_test
    .groupby("Variant")
    .agg(
        Users=("User_ID", "count"),
        Conversions=("Converted", "sum")
    )
    .reset_index()
)

ab_summary["Conversion_Rate"] = (
    ab_summary["Conversions"] / ab_summary["Users"]
)

# ==========================================
# PDF BUILDER
# Parses markdown headings from Gemini's
# response and maps them to proper PDF styles
# ==========================================

def build_pdf(report_text, output_path):
    doc    = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    # Custom styles for heading levels
    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor("#1a237e")
    )

    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
        textColor=colors.HexColor("#283593")
    )

    h3_style = ParagraphStyle(
        "CustomH3",
        parent=styles["Heading3"],
        fontSize=12,
        spaceAfter=6,
        textColor=colors.HexColor("#3949ab")
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        spaceAfter=4,
        leading=14
    )

    elements = []

    # Title
    elements.append(
        Paragraph("Marketing Executive Report", h1_style)
    )
    elements.append(Spacer(1, 0.5*cm))

    # Parse each line and map to correct style
    for line in report_text.split("\n"):
        stripped = line.strip()

        if not stripped:
            elements.append(Spacer(1, 0.2*cm))
            continue

        # H1: lines starting with # (not ##)
        if re.match(r"^# [^#]", stripped):
            text = stripped.lstrip("# ").strip()
            elements.append(Paragraph(text, h1_style))
            elements.append(Spacer(1, 0.3*cm))

        # H2: lines starting with ##
        elif re.match(r"^## [^#]", stripped):
            text = stripped.lstrip("# ").strip()
            elements.append(Paragraph(text, h2_style))
            elements.append(Spacer(1, 0.2*cm))

        # H3: lines starting with ###
        elif re.match(r"^### ", stripped):
            text = stripped.lstrip("# ").strip()
            elements.append(Paragraph(text, h3_style))
            elements.append(Spacer(1, 0.15*cm))

        # Numbered section headings e.g. "1. Executive Summary"
        elif re.match(r"^\d+\.", stripped):
            elements.append(Paragraph(stripped, h2_style))
            elements.append(Spacer(1, 0.2*cm))

        # Bold lines (Gemini often uses **text**)
        elif stripped.startswith("**") and stripped.endswith("**"):
            text = stripped.strip("*")
            elements.append(
                Paragraph(f"<b>{text}</b>", body_style)
            )

        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("• "):
            text = stripped.lstrip("-• ").strip()
            elements.append(
                Paragraph(f"• {text}", body_style)
            )

        else:
            # Replace markdown bold **text** inline
            cleaned = re.sub(
                r"\*\*(.+?)\*\*",
                r"<b>\1</b>",
                stripped
            )
            elements.append(Paragraph(cleaned, body_style))

    doc.build(elements)

# ==========================================
# GENERATE REPORT
# ==========================================

if "exec_report_text" not in st.session_state:
    st.session_state.exec_report_text = None

if st.button("Generate Executive Report", type="primary"):

    with st.spinner("Generating Executive Insights..."):

        prompt = f"""
You are a VP of Marketing Analytics.

Create a professional executive report with clear markdown headings.

Use ## for section headings and ### for sub-headings.

Include:

## 1. Executive Summary
## 2. Business Performance Overview
## 3. Campaign Analysis
## 4. Budget Allocation Recommendations
## 5. Search Query Intelligence
## 6. Keyword Opportunities
## 7. AI Search Visibility Analysis
## 8. Experimentation Insights
## 9. Key Risks
## 10. Growth Opportunities
## 11. Next Quarter Action Plan
## 12. Estimated Business Impact

CAMPAIGN SUMMARY:
{campaign_summary.to_string(index=False)}

TOP SEARCH QUERIES:
{top_queries.to_string(index=False)}

TOP KEYWORDS:
{top_keywords.to_string(index=False)}

AI VISIBILITY:
{visibility_summary.to_string(index=False)}

AB TEST RESULTS:
{ab_summary.to_string(index=False)}
"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.session_state.exec_report_text = response.text
            st.success("Executive Report Generated")

        except Exception as e:
            st.error(str(e))

# ==========================================
# DISPLAY + DOWNLOAD
# ==========================================

if st.session_state.exec_report_text:

    report_text = st.session_state.exec_report_text

    st.subheader("Executive Report")
    st.markdown(report_text)

    # Build PDF with proper heading hierarchy
    pdf_path = "/tmp/Executive_Report.pdf"

    try:
        build_pdf(report_text, pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name="Executive_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"PDF generation failed: {e}")

