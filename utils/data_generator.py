import pandas as pd
import numpy as np
import plotly.express as px

np.random.seed(42)

n = 5000

campaigns = pd.DataFrame({
    "Campaign_ID": range(1,n+1),
    "Campaign_Type": np.random.choice(
        ["Brand","Generic","Competitor"],
        n
    ),
    "Device": np.random.choice(
        ["Mobile","Desktop","Tablet"],
        n
    ),
    "Impressions": np.random.randint(
        1000,
        100000,
        n
    ),
    "Clicks": np.random.randint(
        100,
        10000,
        n
    ),
    "Avg_CPC": np.round(
        np.random.uniform(
            5,
            50,
            n
        ),
        2
    )
})

campaigns["Spend"] = (
    campaigns["Clicks"]
    *
    campaigns["Avg_CPC"]
)

campaigns["CTR"] = (
    campaigns["Clicks"]
    /
    campaigns["Impressions"]
)

campaigns["CVR"] = np.round(
    np.random.uniform(
        0.02,
        0.25,
        n
    ),
    3
)

campaigns["Conversions"] = (
    campaigns["Clicks"]
    *
    campaigns["CVR"]
).astype(int)

campaigns["Revenue_Per_Conversion"] = np.random.randint(
    1000,
    5000,
    n
)

campaigns["Revenue"] = (
    campaigns["Conversions"]
    *
    campaigns["Revenue_Per_Conversion"]
)

campaigns["CPA"] = (
    campaigns["Spend"]
    /
    campaigns["Conversions"]
)

campaigns["ROAS"] = (
    campaigns["Revenue"]
    /
    campaigns["Spend"]
)

campaigns["Customer_Quality_Score"] = ((campaigns["ROAS"] * 20)
    + (campaigns["CVR"] * 200)
    + np.random.normal(
        0,
        10,
        n
    ))

campaigns["Customer_Quality_Score"] = (
    campaigns["Customer_Quality_Score"]
    .clip(
        lower=1,
        upper=100
    ))

campaigns["Customer_LTV"] = (
    campaigns["Revenue"]
    * np.random.uniform(
        1.5,
        5,
        n
    )).astype(int)

campaigns.to_csv(
    "datasets/campaigns.csv",
    index=False
)
print(campaigns.columns)
print("Campaign dataset created")

keywords = pd.DataFrame({

    "Keyword":[
        f"keyword_{i}"
        for i in range(5000)
    ],

    "Quality_Score": np.random.randint(
        1,
        11,
        5000
    ),

    "Impressions": np.random.randint(
        500,
        100000,
        5000
    )
})

keywords["Clicks"] = (
    keywords["Impressions"]
    *
    np.random.uniform(
        0.01,
        0.15,
        5000
    )
).astype(int)

keywords["CTR"] = (
    keywords["Clicks"]
    /
    keywords["Impressions"]
)

keywords["Conversions"] = (
    keywords["Clicks"]
    *
    np.random.uniform(
        0.02,
        0.20,
        5000
    )
).astype(int)

keywords["CVR"] = (
    keywords["Conversions"]
    /
    keywords["Clicks"]
)

keywords["Avg_CPC"] = np.round(
    np.random.uniform(
        5,
        50,
        5000
    ),
    2
)

keywords["Spend"] = (
    keywords["Clicks"]
    *
    keywords["Avg_CPC"]
)

keywords.to_csv(
    "datasets/keywords.csv",
    index=False
)

print("Keyword dataset created")

themes = {
    "Credit Cards":[
        "business credit card",
        "corporate credit card",
        "small business credit card",
        "startup credit card",
        "company credit card",
        "commercial credit card"
    ],

    "Travel":[
        "travel rewards card",
        "airline rewards card",
        "hotel rewards card",
        "corporate travel card",
        "business travel credit card"
    ],

    "Cashback":[
        "cashback business card",
        "cash rewards card",
        "business cashback card",
        "cashback credit card"
    ],

    "Expense Management":[
        "expense management card",
        "corporate expense card",
        "employee expense card",
        "expense tracking software",
        "spend management platform"
    ],

    "Business Loans":[
        "small business loan",
        "startup loan",
        "working capital loan",
        "business financing",
        "merchant cash advance"
    ],

    "Brand":[
        "american express business card",
        "amex corporate card",
        "amex rewards card",
        "american express travel card"
    ]
}

modifiers = [
    "best",
    "top",
    "compare",
    "apply for",
    "cheap",
    "premium",
    "high limit",
    "low fee",
    "instant approval",
    "for startups",
    "for small business",
    "for freelancers",
    "for ecommerce",
    "for travel",
    "for employees",
    "with cashback",
    "with rewards",
    "with airport lounge access",
    "2026"
]

queries_list = []

for _ in range(5000):
    theme = np.random.choice(
        list(themes.keys())
    )
    query = (
        np.random.choice(modifiers)
        + " "
        + np.random.choice(
            themes[theme]
        ))
    queries_list.append(query)

queries = pd.DataFrame({
    "Query": queries_list,
    "Clicks": np.random.randint(
        10,
        500,
        5000)
})

print(
    "Unique Queries:",
    queries["Query"].nunique()
)

queries["Conversions"] = (
    queries["Clicks"]
    *
    np.random.uniform(
        0.01,
        0.15,
        5000
    )
).astype(int)

queries["Avg_Position"] = np.round(
    np.random.uniform(
        1,
        10,
        5000
    ),
    2
)

queries["Quality_Score"] = np.random.randint(
    1,
    11,
    5000
)

# CPC and Spend
queries["Avg_CPC"] = np.round(
    np.random.uniform(
        5,
        50,
        5000
    ),
    2
)

queries["Spend"] = (
    queries["Clicks"]
    *
    queries["Avg_CPC"]
)

# Revenue Columns
queries["Revenue_Per_Conversion"] = np.random.randint(
    1000,
    10000,
    5000
)

queries["Revenue"] = (
    queries["Conversions"]
    *
    queries["Revenue_Per_Conversion"]
)

queries["CPA"] = (
    queries["Spend"] /
    queries["Conversions"].replace(0, 1)
)

queries["ROAS"] = (
    queries["Revenue"] /
    queries["Spend"]
)

queries["CVR"] = (
    queries["Conversions"] /
    queries["Clicks"]
)

queries.to_csv(
    "datasets/search_queries.csv",
    index=False
)

print("Search query dataset created")

users = np.random.randint(
    5000,
    10000,
    5000
)

landing_visits = (
    users
    * np.random.uniform(
        0.4,
        0.8,
        5000
    )
).astype(int)

leads = (
    landing_visits
    * np.random.uniform(
        0.1,
        0.5,
        5000
    )
).astype(int)

conversions = (
    leads
    * np.random.uniform(
        0.05,
        0.4,
        5000
    )
).astype(int)

conversion = pd.DataFrame({
    "Users": users,
    "Landing_Page_Visits": landing_visits,
    "Leads": leads,
    "Conversions": conversions
})

conversion.to_csv(
    "datasets/conversions.csv",
    index=False
)

print("Conversion dataset created")

# ==========================================
# Multi-Touch Attribution Dataset
# ==========================================

channels = [
    "Paid Search",
    "Display",
    "Email",
    "Organic Search",
    "Social Media"
]

journeys = []

for customer_id in range(1, 5001):

    n_touches = np.random.randint(2, 6)

    touchpoints = list(
        np.random.choice(
            channels,
            n_touches,
            replace=True
        )
    )

    conversion_value = np.random.randint(
        1000,
        10000
    )

    journeys.append({
        "Customer_ID": customer_id,
        "Journey": " > ".join(touchpoints),
        "Conversion_Value": conversion_value
    })

attribution_df = pd.DataFrame(
    journeys
)

attribution_df.to_csv(
    "datasets/customer_journeys.csv",
    index=False
)

print(
    "Customer Journey Dataset Created"
)

# Bayesian A/B Testing Dataset
n_users = 10000

ab_test = pd.DataFrame({

    "User_ID": range(
        1,
        n_users + 1
    ),

    "Variant": np.random.choice(
        ["A", "B"],
        n_users
    )
})

ab_test["Converted"] = np.where(
    (
        (ab_test["Variant"] == "A")
        &
        (np.random.rand(n_users) < 0.08)
    )
    |
    (
        (ab_test["Variant"] == "B")
        &
        (np.random.rand(n_users) < 0.10)
    ),
    1,
    0
)

ab_test["Revenue"] = np.where(
    ab_test["Converted"] == 1,
    np.random.randint(
        1000,
        10000,
        n_users
    ),
    0
)

ab_test.to_csv(
    "datasets/ab_test_data.csv",
    index=False
)

print(
    "AB Test Dataset Created"
)

# AI Search Visibility Dataset

platforms = [
    "ChatGPT",
    "Gemini",
    "Perplexity",
    "Google AI Overview"
]

brands = [
    "American Express",
    "Chase",
    "Capital One",
    "Citi",
    "Discover"
]

prompts = [
    "best business credit card",
    "best travel rewards card",
    "best corporate card",
    "best cashback business card",
    "best expense management card",
    "top business credit card for startups",
    "best premium business card",
    "best company expense card"
]

visibility_records = []

for _ in range(5000):

    platform = np.random.choice(platforms)

    prompt = np.random.choice(prompts)

    mentioned_brands = np.random.choice(
        brands,
        size=np.random.randint(1,4),
        replace=False
    )

    rank = np.random.randint(
        1,
        len(mentioned_brands)+1
    )

    visibility_records.append({

        "Platform": platform,

        "Prompt": prompt,

        "Brand_Mentioned":
        mentioned_brands[0],

        "Rank": rank
    })

visibility_df = pd.DataFrame(
    visibility_records
)

visibility_df.to_csv(
    "datasets/ai_visibility.csv",
    index=False
)

print(
    "AI Visibility Dataset Created"
)