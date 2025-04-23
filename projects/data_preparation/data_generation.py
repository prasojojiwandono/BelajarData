from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta

fake = Faker()

# Helper function to generate a random date within a range
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Generate Sales Data
sales_data = []
for _ in range(10000):
    sales_data.append({
        "Date": random_date(datetime(2023, 1, 1), datetime(2023, 12, 31)).strftime("%Y-%m-%d"),
        "Customer": fake.name(),
        "Region": fake.state(),
        "Product": random.choice(["Laptop", "Smartphone", "Tablet", "Monitor", "Keyboard"]),
        "Units Sold": random.randint(1, 10),
        "Unit Price": round(random.uniform(50, 1500), 2)
    })
sales_df = pd.DataFrame(sales_data)
sales_df["Total Sales"] = sales_df["Units Sold"] * sales_df["Unit Price"]
sales_df.to_csv("sales_data.csv", index=False)

# Generate Marketing Data
marketing_data = []
for _ in range(100):
    start_date = random_date(datetime(2023, 1, 1), datetime(2023, 12, 1))
    end_date = start_date + timedelta(days=random.randint(7, 30))
    marketing_data.append({
        "Campaign Name": fake.catch_phrase(),
        "Channel": random.choice(["Email", "Social Media", "TV", "Billboard", "Online Ads"]),
        "Start Date": start_date.strftime("%Y-%m-%d"),
        "End Date": end_date.strftime("%Y-%m-%d"),
        "Budget ($)": random.randint(1000, 20000),
        "Leads Generated": random.randint(50, 1000),
        "Conversions": random.randint(10, 500)
    })
marketing_df = pd.DataFrame(marketing_data)
marketing_df.to_csv("marketing_data.csv", index=False)

# Generate Social Media Data
social_data = []
for _ in range(100):
    social_data.append({
        "Platform": random.choice(["Facebook", "Instagram", "Twitter", "LinkedIn", "TikTok"]),
        "Post Date": random_date(datetime(2023, 1, 1), datetime(2023, 12, 31)).strftime("%Y-%m-%d"),
        "Post Type": random.choice(["Image", "Video", "Text", "Link"]),
        "Reach": random.randint(1000, 50000),
        "Engagement": random.randint(100, 5000),
        "Clicks": random.randint(10, 1000),
        "Shares": random.randint(5, 500)
    })
social_df = pd.DataFrame(social_data)
social_df.to_csv("social_media_data.csv", index=False)

# Generate Finance Data
finance_data = []
for _ in range(100):
    finance_data.append({
        "Date": random_date(datetime(2023, 1, 1), datetime(2023, 12, 31)).strftime("%Y-%m-%d"),
        "Account": random.choice(["Revenue", "Expenses", "Assets", "Liabilities", "Equity"]),
        "Category": random.choice(["Salary", "Rent", "Sales", "Utilities", "Investment"]),
        "Amount ($)": round(random.uniform(-10000, 20000), 2)
    })
finance_df = pd.DataFrame(finance_data)
finance_df.to_csv("finance_data.csv", index=False)

# Generate SEO Data
seo_data = []
for _ in range(100):
    seo_data.append({
        "Date": random_date(datetime(2023, 1, 1), datetime(2023, 12, 31)).strftime("%Y-%m-%d"),
        "Keyword": fake.word(),
        "Search Volume": random.randint(100, 10000),
        "Clicks": random.randint(10, 5000),
        "CTR (%)": round(random.uniform(1.0, 20.0), 2),
        "Avg. Position": round(random.uniform(1.0, 50.0), 2)
    })
seo_df = pd.DataFrame(seo_data)
seo_df.to_csv("seo_data.csv", index=False)

print("All CSV files generated successfully!")
