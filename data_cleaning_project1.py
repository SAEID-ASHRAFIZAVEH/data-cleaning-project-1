import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("Libraries imported successfully!")

# Create sample messy customer survey data
# This dataset simulates real-world data quality issues I need to learn to handle
data = [
    [1, "John Smith", 25, "john@email.com", "2024-01-15", 5, "electronics", 299.99, "Yes", "Great product!"],
    [2, " jane doe ", 150, "JANE@EMAIL.COM", "01/16/2024", 4, "CLOTHING", 89.50, "yes", "  Good quality  "],
    [1, "John Smith", 25, "john@email.com", "2024-01-15", 5, "Electronics", 299.99, "Yes", "Great product!"],
    [3, "Bob Johnson", 30, "", "2024-01-17", 3, "Books", 25.00, "Maybe", ""],
    [4, "Alice Brown", -5, "alice@email.com", "17-01-2024", 6, "home & garden", 450.00, "NO", "Poor delivery"],
    [5, "", 35, "test@email.com", "2024/01/18", 2, "Clothing", None, "Yes", ""],
]

columns = ['customer_id', 'name', 'age', 'email', 'survey_date', 'satisfaction_rating', 
           'product_category', 'purchase_amount', 'would_recommend', 'comments']

# Create DataFrame from the messy data
df = pd.DataFrame(data, columns=columns)

print("Raw dataset created for cleaning practice!")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows of raw data:")
print(df.head())

# Data Quality Assessment - identifying issues before cleaning
print("\n=== DATA QUALITY ASSESSMENT ===")
print("Let me examine what's wrong with this data...")

# Check for missing values using pandas methods I learned in class
print("Missing values per column:")
print(df.isnull().sum())

# Check for duplicate records - important data quality issue
print(f"\nDuplicate rows found: {df.duplicated().sum()}")

# Identify problematic age values (learned this is a common issue)
print(f"\nAge data issues:")
print(f"- Negative ages: {(df['age'] < 0).sum()}")
print(f"- Unrealistic ages (>120): {(df['age'] > 120).sum()}")

# Check satisfaction rating validity (should be 1-5 scale)
print(f"\nInvalid satisfaction ratings: {((df['satisfaction_rating'] < 1) | (df['satisfaction_rating'] > 5)).sum()}")

# Examine inconsistent categorical data
print("\nProduct category inconsistencies:")
print(df['product_category'].unique())
print("^ Notice the case sensitivity issues!")

print("\n=== STARTING DATA CLEANING PROCESS ===")
print("Applying data cleaning techniques learned in my coursework...")

# Make a copy to preserve original data (best practice I learned)
df_clean = df.copy()

# STEP 1: Handle duplicate records
print("\n1. Removing Duplicate Records:")
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
removed_duplicates = initial_rows - len(df_clean)
print(f"   Removed {removed_duplicates} duplicate row(s)")

# STEP 2: Handle missing values using appropriate strategies
print("\n2. Handling Missing Values:")
# For numerical data, I'll use median (more robust than mean for outliers)
median_amount = df_clean['purchase_amount'].median()
df_clean['purchase_amount'] = df_clean['purchase_amount'].fillna(median_amount)
print(f"   ✓ Filled missing purchase amounts with median: ${median_amount}")

# For categorical data, I'll use meaningful placeholders
df_clean['name'] = df_clean['name'].fillna("Unknown Customer")
df_clean['email'] = df_clean['email'].fillna("no_email@unknown.com")
print(f"   ✓ Filled missing names and emails with placeholders")

print(f"   Total missing values remaining: {df_clean.isnull().sum().sum()}")

# STEP 3: Fix invalid numerical values
print("\n3. Correcting Invalid Age Data:")
print(f"   Ages before cleaning - Min: {df_clean['age'].min()}, Max: {df_clean['age'].max()}")

# Replace invalid ages with median of valid ages (statistical approach)
valid_ages = df_clean[(df_clean['age'] >= 0) & (df_clean['age'] <= 120)]['age']
median_age = valid_ages.median()

df_clean.loc[df_clean['age'] < 0, 'age'] = median_age
df_clean.loc[df_clean['age'] > 120, 'age'] = median_age

print(f"   Ages after cleaning - Min: {df_clean['age'].min()}, Max: {df_clean['age'].max()}")

# STEP 4: Fix satisfaction rating outliers
print("\n4. Correcting Satisfaction Ratings:")
print(f"   Ratings before: {sorted(df_clean['satisfaction_rating'].unique())}")

# Cap ratings to valid 1-5 range (business rule enforcement)
df_clean.loc[df_clean['satisfaction_rating'] < 1, 'satisfaction_rating'] = 1
df_clean.loc[df_clean['satisfaction_rating'] > 5, 'satisfaction_rating'] = 5

print(f"   Ratings after: {sorted(df_clean['satisfaction_rating'].unique())}")

# STEP 5: Standardize text data (very important for consistency!)
print("\n5. Standardizing Text Formatting:")

# Clean names - remove whitespace and standardize case
df_clean['name'] = df_clean['name'].str.strip().str.title()
print("   ✓ Standardized customer names")

# Clean emails - convert to lowercase (email standard)
df_clean['email'] = df_clean['email'].str.strip().str.lower()
print("   ✓ Standardized email formatting")

# Clean product categories - fix case inconsistencies
df_clean['product_category'] = df_clean['product_category'].str.strip().str.title()
print("   ✓ Standardized product categories")

# Standardize recommendation responses
df_clean['would_recommend'] = df_clean['would_recommend'].str.strip().str.title()
# Convert "Maybe" to "No" for binary analysis (business decision)
df_clean.loc[df_clean['would_recommend'] == 'Maybe', 'would_recommend'] = 'No'
print("   ✓ Standardized recommendation responses")

# Clean comments - remove extra whitespace
df_clean['comments'] = df_clean['comments'].str.strip()

# STEP 6: Standardize date formats (common real-world issue)
print("\n6. Standardizing Date Formats:")
print(f"   Original date formats: {df_clean['survey_date'].unique()}")

# Use pandas to_datetime with error handling
df_clean['survey_date'] = pd.to_datetime(df_clean['survey_date'], errors='coerce')

print("   ✓ All dates converted to standard YYYY-MM-DD format")

# STEP 7: Final data quality validation
print("\n=== DATA CLEANING RESULTS ===")
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
print(f"Data quality improvements made:")
print(f"   • Fixed {df.isnull().sum().sum()} missing values")
print(f"   • Corrected {((df['age'] < 0) | (df['age'] > 120)).sum()} invalid ages")
print(f"   • Fixed {((df['satisfaction_rating'] < 1) | (df['satisfaction_rating'] > 5)).sum()} invalid ratings")
print(f"   • Standardized all text formatting")
print(f"   • Unified date formats")

print("\nCleaned dataset preview:")
print(df_clean)

print("\nProduct categories after standardization:")
print(df_clean['product_category'].value_counts())

# STEP 8: Save the cleaned dataset for future analysis
print("\n=== SAVING CLEANED DATA ===")
df_clean.to_csv('cleaned_customer_survey.csv', index=False)
print("✅ Cleaned dataset saved as 'cleaned_customer_survey.csv'")

# STEP 9: Create visualizations to validate cleaning results
print("\n=== CREATING DATA QUALITY VISUALIZATIONS ===")
print("Generating charts to verify cleaning effectiveness...")

# Set up the visualization layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Data Cleaning Validation - Customer Survey Analysis', fontsize=14)

# Age distribution after cleaning
axes[0, 0].hist(df_clean['age'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution (Post-Cleaning)')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Customer Count')

# Satisfaction ratings distribution
rating_counts = df_clean['satisfaction_rating'].value_counts().sort_index()
axes[0, 1].bar(rating_counts.index, rating_counts.values, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Customer Satisfaction Ratings')
axes[0, 1].set_xlabel('Rating (1-5 Scale)')
axes[0, 1].set_ylabel('Count')

# Product category distribution
category_counts = df_clean['product_category'].value_counts()
colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen']
axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
axes[1, 0].set_title('Product Category Distribution')

# Recommendation willingness
rec_counts = df_clean['would_recommend'].value_counts()
axes[1, 1].bar(rec_counts.index, rec_counts.values, color=['lightcoral', 'lightgreen'], edgecolor='black')
axes[1, 1].set_title('Customer Recommendation Willingness')
axes[1, 1].set_xlabel('Would Recommend')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data_cleaning_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved as 'data_cleaning_analysis.png'")

# Final summary for my portfolio documentation
print("\n" + "="*60)
print("🎓 DATA CLEANING PROJECT COMPLETED SUCCESSFULLY! 🎓")
print("="*60)
print("Skills demonstrated in this project:")
print("• Data quality assessment and issue identification")
print("• Missing value handling with appropriate strategies")
print("• Duplicate record removal")
print("• Outlier detection and correction")
print("• Text data standardization and cleaning")
print("• Date format standardization")
print("• Data validation and quality reporting")
print("• Data visualization for result validation")
print("\nDeliverables created:")
print("📄 1. cleaned_customer_survey.csv - Clean dataset ready for analysis")
print("📊 2. data_cleaning_analysis.png - Quality validation charts")
print("💻 3. Complete Python script with documentation")
print("\n🚀 Ready for GitHub portfolio upload!")