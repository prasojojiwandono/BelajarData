# 📦 1. Setup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# 📥 2. Load Data
df = pd.read_csv("wishlist_flat.csv")  # Make sure this is pre-exploded per kategori

# 🧼 3. Clean & Normalize
df['kategori'] = df['kategori'].str.strip().str.lower()
df['gender'] = df['gender'].str.title().fillna('Unknown')

# 🧽 4. Remove rare/irrelevant categories (including "luxury")
min_category_support = 10
valid_categories = df['kategori'].value_counts()
valid_categories = valid_categories[valid_categories >= min_category_support].index
df = df[df['kategori'].isin(valid_categories)]

# 🧼 5. Remove users with < 3 wishlist entries
wishlist_counts = df.groupby('user_id').size()
valid_users = wishlist_counts[wishlist_counts >= 3].index
df = df[df['user_id'].isin(valid_users)]

# 🧮 6. Get most preferred category per user
user_cat_counts = (
    df.groupby(['user_id', 'kategori'])
    .size()
    .reset_index(name='count')
)
top_cat = (
    user_cat_counts
    .sort_values(['user_id', 'count'], ascending=[True, False])
    .drop_duplicates('user_id')
    .rename(columns={'kategori': 'target_category'})
)

# 👤 7. Get user features
user_features = df.groupby('user_id')[['gender', 'age']].first().reset_index()
user_features['gender'] = user_features['gender'].map({'Male': 0, 'Female': 1, 'Unknown': -1})
user_features = user_features.dropna(subset=['age'])

# 🔗 8. Merge features with target labels
train_df = user_features.merge(top_cat[['user_id', 'target_category']], on='user_id')

# 🎯 9. Encode labels
label_encoder = LabelEncoder()
train_df['target_encoded'] = label_encoder.fit_transform(train_df['target_category'])

# ⚠️ 10. Final filtering for categories with enough users
valid_cats = train_df['target_category'].value_counts()
valid_cats = valid_cats[valid_cats >= 2].index
train_df_filtered = train_df[train_df['target_category'].isin(valid_cats)].copy()

# 🚨 11. Check sample size
if len(train_df_filtered) < 5:
    print("❌ Data terlalu sedikit. Tambah data atau perpanjang periode.")
else:
    # 🧪 12. Train/Test Split
    X = train_df_filtered[['gender', 'age']]
    y = train_df_filtered['target_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 🌳 13. Train Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 📊 14. Evaluation
    y_pred = clf.predict(X_test)
    actual_labels = unique_labels(y_test, y_pred)
    target_names = label_encoder.inverse_transform(actual_labels)
    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 🔮 15. Predict top-N categories per user
    def predict_top_categories(gender, age, top_n=3):
        g = {'Male': 0, 'Female': 1, 'Unknown': -1}.get(gender.title(), -1)
        probs = clf.predict_proba([[g, age]])[0]
        top_indices = probs.argsort()[::-1][:top_n]
        return label_encoder.inverse_transform(top_indices)

    # 📦 16. Output: Age + Predicted Categories
    unique_ages = sorted(train_df_filtered['age'].unique())
    output = []
    for age in sorted(train_df_filtered['age'].unique()):
        for gender in ['Male', 'Female', 'Unknown']:
            top_preds = predict_top_categories(gender, age, top_n=3)
            output.append({
            'age': int(age),
            'gender': gender,
            'predicted_top_categories': ', '.join(top_preds)
        })


    output_df = pd.DataFrame(output)
    output_df.to_csv("predicted_age_top_categories.csv", index=False)
    print("✅ Output saved to 'predicted_age_top_categories.csv'")
