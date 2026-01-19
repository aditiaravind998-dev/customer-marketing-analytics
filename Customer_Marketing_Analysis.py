#!/usr/bin/env python
# coding: utf-8

# # Customer Marketing Data Analysis for FMCG Retail Company
# ## A Comprehensive Analysis Using Python, Machine Learning, and Marketing Strategy Development

# ## 1. Import Libraries and Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_excel('/mnt/user-data/uploads/Customer_marketing_dataset.xlsx')
print(f"Dataset loaded: {df.shape[0]} customers, {df.shape[1]} features")

# ## 2. Data Preprocessing and Feature Engineering

# ### 2.1 Handle Missing Values
print(f"\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill missing Income values with median (robust to outliers)
df['Income'].fillna(df['Income'].median(), inplace=True)

# Remove outliers in Income (666666 is clearly an error)
df = df[df['Income'] < 200000]
print(f"\nDataset after removing outliers: {df.shape[0]} customers")

# ### 2.2 Feature Engineering

# Calculate Age from Year_Birth (using 2024 as reference)
df['Age'] = 2024 - df['Year_Birth']

# Remove unrealistic ages (people born in 1893 would be 131 years old)
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
print(f"Dataset after age filtering: {df.shape[0]} customers")

# Create Age Groups
def age_group(age):
    if age < 30:
        return 'Young Adult (18-29)'
    elif age < 45:
        return 'Middle-Aged (30-44)'
    elif age < 60:
        return 'Senior (45-59)'
    else:
        return 'Elderly (60+)'

df['Age_Group'] = df['Age'].apply(age_group)

# Total spending across all product categories
df['Total_Spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts']

# Total number of purchases across all channels
df['Total_Purchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']

# Total number of children (kids + teens)
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Has children flag
df['Has_Children'] = (df['Total_Children'] > 0).astype(int)

# Total campaigns accepted
df['Total_Campaigns_Accepted'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['Response']

# Customer tenure (days since enrollment - assuming Dt_Customer is the enrollment date)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
reference_date = df['Dt_Customer'].max()
df['Customer_Tenure_Days'] = (reference_date - df['Dt_Customer']).dt.days

# Average spending per purchase
df['Avg_Spending_Per_Purchase'] = df['Total_Spending'] / (df['Total_Purchases'] + 1)

# Web engagement ratio
df['Web_Engagement_Ratio'] = df['NumWebPurchases'] / (df['Total_Purchases'] + 1)

# Deal sensitivity (proportion of purchases made with deals)
df['Deal_Sensitivity'] = df['NumDealsPurchases'] / (df['Total_Purchases'] + df['NumDealsPurchases'] + 1)

# Clean Marital Status (consolidate similar categories)
marital_mapping = {
    'Single': 'Single',
    'Together': 'Partner',
    'Married': 'Partner',
    'Divorced': 'Single',
    'Widow': 'Single',
    'Alone': 'Single',
    'Absurd': 'Other',
    'YOLO': 'Other'
}
df['Marital_Status_Clean'] = df['Marital_Status'].map(marital_mapping)

# Education level encoding
education_order = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
df['Education_Level'] = df['Education'].map(education_order)

print(f"\nFinal dataset: {df.shape[0]} customers")
print(f"Total features after engineering: {df.shape[1]}")

# ## 3. Exploratory Data Analysis (EDA)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Age Distribution
axes[0, 0].hist(df['Age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution of Customers')
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
axes[0, 0].legend()

# Income Distribution
axes[0, 1].hist(df['Income'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Income ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Income Distribution of Customers')
axes[0, 1].axvline(df['Income'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Income"].mean():,.0f}')
axes[0, 1].legend()

# Total Spending Distribution
axes[0, 2].hist(df['Total_Spending'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[0, 2].set_xlabel('Total Spending ($)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Total Spending Distribution')

# Spending by Product Category
spending_categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']
spending_means = [df[col].mean() for col in spending_categories]
labels = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets']
colors = plt.cm.Set3(np.linspace(0, 1, 5))
axes[1, 0].bar(labels, spending_means, color=colors, edgecolor='black')
axes[1, 0].set_xlabel('Product Category')
axes[1, 0].set_ylabel('Mean Spending ($)')
axes[1, 0].set_title('Average Spending by Product Category')

# Purchase Channel Distribution
channels = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
channel_means = [df[col].mean() for col in channels]
channel_labels = ['Web', 'Catalog', 'Store']
axes[1, 1].bar(channel_labels, channel_means, color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
axes[1, 1].set_xlabel('Purchase Channel')
axes[1, 1].set_ylabel('Mean Number of Purchases')
axes[1, 1].set_title('Average Purchases by Channel')

# Campaign Response Rates
campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'Response']
campaign_rates = [df[col].mean()*100 for col in campaigns]
campaign_labels = ['Campaign 1', 'Campaign 2', 'Campaign 3', 'Last Campaign']
axes[1, 2].bar(campaign_labels, campaign_rates, color='#9b59b6', edgecolor='black')
axes[1, 2].set_xlabel('Campaign')
axes[1, 2].set_ylabel('Response Rate (%)')
axes[1, 2].set_title('Campaign Response Rates')

plt.tight_layout()
plt.savefig('/home/claude/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA visualization saved: eda_overview.png")

# ## 4. Correlation Analysis

# Select numerical features for correlation
numerical_features = ['Income', 'Age', 'Total_Spending', 'Total_Purchases', 'Total_Children',
                     'Recency', 'NumWebVisitsMonth', 'Deal_Sensitivity', 'Customer_Tenure_Days',
                     'MntWines', 'MntMeatProducts', 'NumWebPurchases', 'NumStorePurchases']

correlation_matrix = df[numerical_features].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Key Customer Features', fontsize=14)
plt.tight_layout()
plt.savefig('/home/claude/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Correlation matrix saved: correlation_matrix.png")

# Key correlations
print("\n=== Key Correlations with Total Spending ===")
spending_corr = correlation_matrix['Total_Spending'].sort_values(ascending=False)
print(spending_corr)

# ## 5. Customer Segmentation using K-Means Clustering

# ### 5.1 Prepare Features for Clustering

# RFM-based segmentation features
clustering_features = ['Recency', 'Total_Purchases', 'Total_Spending', 'Income', 
                       'Age', 'Deal_Sensitivity', 'Web_Engagement_Ratio']

X_cluster = df[clustering_features].copy()

# Standardize features (critical for K-Means as it uses Euclidean distance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("\n=== K-Means Clustering Mathematics ===")
print("""
K-Means Algorithm:
1. Objective: Minimize Within-Cluster Sum of Squares (WCSS)
   J = Σ(i=1 to K) Σ(x∈Ci) ||x - μi||²
   
   Where:
   - K = number of clusters
   - Ci = set of points in cluster i
   - μi = centroid of cluster i
   - ||x - μi||² = squared Euclidean distance

2. Algorithm Steps:
   a) Initialize K centroids randomly
   b) Assign each point to nearest centroid
   c) Update centroids as mean of assigned points
   d) Repeat until convergence
   
3. Distance Metric (Euclidean):
   d(x, y) = √(Σ(i=1 to n) (xi - yi)²)
""")

# ### 5.2 Determine Optimal Number of Clusters (Elbow Method)

wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
axes[0].set_title('Elbow Method for Optimal K')
axes[0].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[0].legend()

# Silhouette Score
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs Number of Clusters')
axes[1].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/claude/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nElbow and Silhouette analysis saved: elbow_silhouette.png")

# ### 5.3 Apply K-Means with Optimal K=4

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"\nSilhouette Score with K={optimal_k}: {silhouette_score(X_scaled, df['Cluster']):.4f}")

# ### 5.4 Visualize Clusters using PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\n=== PCA Analysis ===")
print(f"Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}")

plt.figure(figsize=(12, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
cluster_names = ['Premium Shoppers', 'Budget Conscious', 'Young Professionals', 'Loyal Traditionalists']

for i in range(optimal_k):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=cluster_names[i], 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# Plot centroids
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', 
            s=200, label='Centroids', edgecolors='white', linewidth=2)

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments Visualized using PCA', fontsize=14)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/home/claude/customer_segments_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("PCA visualization saved: customer_segments_pca.png")

# ### 5.5 Cluster Analysis and Profiling

print("\n=== Customer Segment Profiles ===")
cluster_profile = df.groupby('Cluster').agg({
    'Income': 'mean',
    'Age': 'mean',
    'Total_Spending': 'mean',
    'Total_Purchases': 'mean',
    'Recency': 'mean',
    'Deal_Sensitivity': 'mean',
    'Web_Engagement_Ratio': 'mean',
    'Total_Children': 'mean',
    'Total_Campaigns_Accepted': 'mean',
    'ID': 'count'
}).round(2)

cluster_profile.columns = ['Avg Income', 'Avg Age', 'Avg Spending', 'Avg Purchases', 
                          'Avg Recency', 'Deal Sensitivity', 'Web Ratio', 
                          'Avg Children', 'Campaign Acceptance', 'Customer Count']
print(cluster_profile)

# Visualize cluster profiles
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Spending by Cluster
cluster_spending = df.groupby('Cluster')['Total_Spending'].mean()
axes[0, 0].bar(cluster_names, cluster_spending, color=colors, edgecolor='black')
axes[0, 0].set_xlabel('Customer Segment')
axes[0, 0].set_ylabel('Average Total Spending ($)')
axes[0, 0].set_title('Average Spending by Customer Segment')
axes[0, 0].tick_params(axis='x', rotation=15)

# Income by Cluster
cluster_income = df.groupby('Cluster')['Income'].mean()
axes[0, 1].bar(cluster_names, cluster_income, color=colors, edgecolor='black')
axes[0, 1].set_xlabel('Customer Segment')
axes[0, 1].set_ylabel('Average Income ($)')
axes[0, 1].set_title('Average Income by Customer Segment')
axes[0, 1].tick_params(axis='x', rotation=15)

# Recency by Cluster
cluster_recency = df.groupby('Cluster')['Recency'].mean()
axes[1, 0].bar(cluster_names, cluster_recency, color=colors, edgecolor='black')
axes[1, 0].set_xlabel('Customer Segment')
axes[1, 0].set_ylabel('Average Days Since Last Purchase')
axes[1, 0].set_title('Recency by Customer Segment')
axes[1, 0].tick_params(axis='x', rotation=15)

# Cluster Size
cluster_size = df.groupby('Cluster').size()
axes[1, 1].pie(cluster_size, labels=cluster_names, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=[0.05, 0.05, 0.05, 0.05])
axes[1, 1].set_title('Customer Distribution by Segment')

plt.tight_layout()
plt.savefig('/home/claude/cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Cluster profiles saved: cluster_profiles.png")

# ## 6. Campaign Response Prediction using Machine Learning

# ### 6.1 Prepare Data for Classification

# Target variable: Response (last campaign)
target = 'Response'

# Feature selection for prediction
features_for_prediction = ['Income', 'Age', 'Total_Spending', 'Total_Purchases', 
                          'Recency', 'NumWebVisitsMonth', 'Deal_Sensitivity',
                          'Total_Children', 'Education_Level', 'Customer_Tenure_Days',
                          'MntWines', 'MntMeatProducts', 'NumWebPurchases', 
                          'NumStorePurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3']

X = df[features_for_prediction].copy()
y = df[target]

# Handle any remaining missing values
X.fillna(X.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\n=== Classification Data Preparation ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
print(f"Response rate: {y.mean()*100:.2f}%")

# ### 6.2 Random Forest Classifier

print("\n=== Random Forest Mathematics ===")
print("""
Random Forest Algorithm:
1. Ensemble Learning: Combines multiple decision trees
   - Each tree is trained on a bootstrap sample
   - Random subset of features at each split
   
2. Prediction (Classification):
   ŷ = mode({h1(x), h2(x), ..., hB(x)})
   Where B = number of trees
   
3. Feature Importance (Mean Decrease in Gini):
   Gini Impurity = 1 - Σ(pi²) for all classes i
   
4. Key Parameters:
   - n_estimators: Number of trees
   - max_depth: Maximum tree depth
   - min_samples_split: Minimum samples to split
""")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                  random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nRandom Forest Cross-Validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=['Not Responded', 'Responded']))

# ### 6.3 Gradient Boosting Classifier

print("\n=== Gradient Boosting Mathematics ===")
print("""
Gradient Boosting Algorithm:
1. Sequential Learning: Each tree corrects errors of previous
   
2. Objective: Minimize Loss Function
   L(y, F(x)) = Σ l(yi, F(xi))
   
3. Update Rule:
   F_m(x) = F_{m-1}(x) + η * h_m(x)
   
   Where:
   - η = learning rate
   - h_m = tree fitted to pseudo-residuals
   
4. Pseudo-residuals (Gradient):
   r_im = -[∂L(yi, F(xi)) / ∂F(xi)]
""")

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                      max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

cv_scores_gb = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\nGradient Boosting Cross-Validation AUC: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std()*2:.4f})")

y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

print("\n=== Gradient Boosting Classification Report ===")
print(classification_report(y_test, y_pred_gb, target_names=['Not Responded', 'Responded']))

# ### 6.4 Feature Importance Analysis

feature_importance_rf = pd.DataFrame({
    'Feature': features_for_prediction,
    'Importance_RF': rf_model.feature_importances_,
    'Importance_GB': gb_model.feature_importances_
}).sort_values('Importance_RF', ascending=False)

print("\n=== Top 10 Features for Campaign Response ===")
print(feature_importance_rf.head(10))

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest
top_features = feature_importance_rf.head(10)
axes[0].barh(top_features['Feature'], top_features['Importance_RF'], color='#3498db')
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Random Forest - Top 10 Features')
axes[0].invert_yaxis()

# Gradient Boosting
top_features_gb = feature_importance_rf.sort_values('Importance_GB', ascending=False).head(10)
axes[1].barh(top_features_gb['Feature'], top_features_gb['Importance_GB'], color='#e74c3c')
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Gradient Boosting - Top 10 Features')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/home/claude/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance saved: feature_importance.png")

# ### 6.5 Confusion Matrix Visualization

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Responded', 'Responded'],
            yticklabels=['Not Responded', 'Responded'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Random Forest Confusion Matrix')

# Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=['Not Responded', 'Responded'],
            yticklabels=['Not Responded', 'Responded'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Gradient Boosting Confusion Matrix')

plt.tight_layout()
plt.savefig('/home/claude/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrices saved: confusion_matrices.png")

# ## 7. Channel Analysis and Customer Journey Insights

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Channel preference by Age Group
channel_by_age = df.groupby('Age_Group')[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].mean()
channel_by_age.plot(kind='bar', ax=axes[0, 0], color=['#3498db', '#e74c3c', '#2ecc71'])
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Average Purchases')
axes[0, 0].set_title('Purchase Channel Preference by Age Group')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend(title='Channel')

# Spending by Education Level
education_spending = df.groupby('Education')[['MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 'MntSweetProducts']].mean()
education_spending.plot(kind='bar', ax=axes[0, 1], stacked=True, colormap='Set3')
axes[0, 1].set_xlabel('Education Level')
axes[0, 1].set_ylabel('Average Spending ($)')
axes[0, 1].set_title('Product Spending by Education Level')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(title='Product')

# Web visits vs Purchases (engagement to conversion)
axes[1, 0].scatter(df['NumWebVisitsMonth'], df['NumWebPurchases'], alpha=0.5, c=df['Total_Spending'], cmap='viridis')
axes[1, 0].set_xlabel('Website Visits per Month')
axes[1, 0].set_ylabel('Web Purchases')
axes[1, 0].set_title('Website Engagement vs. Conversion')
cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
cbar.set_label('Total Spending')

# Recency vs Response Rate
recency_bins = pd.cut(df['Recency'], bins=[0, 30, 60, 90, 120], labels=['0-30', '31-60', '61-90', '91-120'])
response_by_recency = df.groupby(recency_bins, observed=True)['Response'].mean() * 100
axes[1, 1].bar(response_by_recency.index.astype(str), response_by_recency.values, color='#9b59b6', edgecolor='black')
axes[1, 1].set_xlabel('Days Since Last Purchase')
axes[1, 1].set_ylabel('Campaign Response Rate (%)')
axes[1, 1].set_title('Recency Impact on Campaign Response')

plt.tight_layout()
plt.savefig('/home/claude/channel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Channel analysis saved: channel_analysis.png")

# ## 8. Summary Statistics for Marketing Strategy

print("\n" + "="*60)
print("SUMMARY: KEY INSIGHTS FOR MARKETING STRATEGY")
print("="*60)

print("\n1. CUSTOMER SEGMENTS (K-Means Clustering):")
for i, name in enumerate(cluster_names):
    cluster_data = df[df['Cluster'] == i]
    print(f"\n   {name} (Cluster {i}):")
    print(f"   - Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"   - Avg Income: ${cluster_data['Income'].mean():,.0f}")
    print(f"   - Avg Spending: ${cluster_data['Total_Spending'].mean():,.0f}")
    print(f"   - Campaign Response: {cluster_data['Response'].mean()*100:.1f}%")

print("\n2. TOP FACTORS FOR CAMPAIGN RESPONSE:")
for idx, row in feature_importance_rf.head(5).iterrows():
    print(f"   - {row['Feature']}: {row['Importance_RF']*100:.1f}%")

print("\n3. CHANNEL INSIGHTS:")
print(f"   - Store purchases dominate: {df['NumStorePurchases'].sum()} total")
print(f"   - Web purchases: {df['NumWebPurchases'].sum()} total")
print(f"   - Catalog purchases: {df['NumCatalogPurchases'].sum()} total")
print(f"   - Avg web visits/month: {df['NumWebVisitsMonth'].mean():.1f}")

print("\n4. PRODUCT INSIGHTS:")
print(f"   - Wines are the highest revenue category: ${df['MntWines'].sum():,}")
print(f"   - Meat products: ${df['MntMeatProducts'].sum():,}")
print(f"   - Premium categories (Wines+Meat) = {(df['MntWines'].sum()+df['MntMeatProducts'].sum())/df['Total_Spending'].sum()*100:.1f}% of total revenue")

print("\n5. CUSTOMER DEMOGRAPHICS:")
print(f"   - Average age: {df['Age'].mean():.0f} years")
print(f"   - Average income: ${df['Income'].mean():,.0f}")
print(f"   - Customers with children: {df['Has_Children'].mean()*100:.1f}%")

print("\n" + "="*60)

# Save processed data
df.to_csv('/home/claude/processed_customer_data.csv', index=False)
print("\nProcessed data saved: processed_customer_data.csv")

print("\n=== Analysis Complete ===")
print("All visualizations saved to /home/claude/")
