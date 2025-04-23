from faker import Faker
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Helper to generate date range
def generate_dates(start, days):
    return [start + timedelta(days=i) for i in range(days)]

# 1. Marketing Manager - E-commerce Sales
def generate_ecommerce_data():
    start_date = datetime(2024, 1, 1)
    days = 365
    dates = generate_dates(start_date, days)

    products = ['T-shirt', 'Shoes', 'Bag', 'Hat', 'Jacket']
    channels = ['Google Ads', 'Instagram', 'Email', 'Direct', 'Facebook']
    campaigns = ['Spring Launch', 'Clearance', 'BOGO', 'Retargeting']

    data = []

    for date in dates:
        for _ in range(random.randint(20, 50)):
            product = random.choice(products)
            channel = random.choice(channels)
            campaign = random.choice(campaigns)
            price = random.randint(20, 120)
            conversion = random.choices([1, 0], weights=[0.3, 0.7])[0]
            data.append({
                'date': date,
                'product': product,
                'channel': channel,
                'campaign': campaign,
                'price': price,
                'converted': conversion
            })
    return pd.DataFrame(data)

# 2. Hospital Admin Data
def generate_hospital_data():
    departments = ['ER', 'Surgery', 'ICU', 'Maternity', 'Pediatrics']
    start_date = datetime(2024, 1, 1)
    days = 30
    dates = generate_dates(start_date, days)

    data = []
    for date in dates:
        for dept in departments:
            data.append({
                'date': date,
                'department': dept,
                'bed_occupied': random.randint(5, 20),
                'surgeries_scheduled': random.randint(2, 10),
                'surgeries_completed': random.randint(1, 9),
                'er_wait_time_min': random.randint(10, 120),
                'satisfaction_score': round(random.uniform(3.5, 5.0), 2)
            })
    return pd.DataFrame(data)

# 3. School Principal Data
def generate_school_data():
    classes = [f"Class {grade}{sec}" for grade in range(7, 13) for sec in ['A', 'B']]
    teachers = [fake.name() for _ in range(10)]
    departments = ['Science', 'Math', 'History', 'English', 'Art']

    start_date = datetime(2024, 1, 1)
    days = 20
    dates = generate_dates(start_date, days)

    attendance_data = []
    for date in dates:
        for class_name in classes:
            attendance_data.append({
                'date': date,
                'class': class_name,
                'attendance_rate': round(random.uniform(0.7, 1.0), 2)
            })

    teacher_data = [{
        'teacher': teacher,
        'knowledge': round(random.uniform(3, 5), 1),
        'communication': round(random.uniform(3, 5), 1),
        'punctuality': round(random.uniform(3, 5), 1),
    } for teacher in teachers]

    budget_data = [{
        'department': dept,
        'budget': random.randint(10000, 30000),
        'spent': random.randint(5000, 28000)
    } for dept in departments]

    return (
        pd.DataFrame(attendance_data),
        pd.DataFrame(teacher_data),
        pd.DataFrame(budget_data)
    )

# 4. Restaurant Owner Data
def generate_restaurant_data():
    dishes = ['Burger', 'Pizza', 'Salad', 'Steak', 'Pasta', 'Soup']
    start_date = datetime(2024, 1, 1)
    days = 30
    dates = generate_dates(start_date, days)

    sales_data = []
    for date in dates:
        for _ in range(random.randint(20, 50)):
            dish = random.choice(dishes)
            price = random.randint(5, 25)
            review = round(random.uniform(3, 5), 1)
            sales_data.append({
                'date': date,
                'dish': dish,
                'price': price,
                'review_score': review
            })

    inventory_data = [{
        'ingredient': ingredient,
        'stock_level': random.randint(10, 100)
    } for ingredient in ['Tomato', 'Cheese', 'Beef', 'Lettuce', 'Flour', 'Onion']]

    return pd.DataFrame(sales_data), pd.DataFrame(inventory_data)

# Example usage
if __name__ == "__main__":
    df_ecom = generate_ecommerce_data()
    df_hosp = generate_hospital_data()
    df_attendance, df_teacher, df_budget = generate_school_data()
    df_sales, df_inventory = generate_restaurant_data()

    # Save to CSV if needed
    df_ecom.to_csv("ecommerce_data.csv", index=False)
    # df_hosp.to_csv("hospital_data.csv", index=False)
    # df_attendance.to_csv("school_attendance.csv", index=False)
    # df_teacher.to_csv("teacher_feedback.csv", index=False)
    # df_budget.to_csv("school_budget.csv", index=False)
    # df_sales.to_csv("restaurant_sales.csv", index=False)
    # df_inventory.to_csv("restaurant_inventory.csv", index=False)

    print("All dummy data generated!")
