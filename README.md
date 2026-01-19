# customer-marketing-analytics
# Customer Marketing Analytics

A comprehensive end-to-end data science project for customer segmentation, campaign response prediction, and marketing strategy optimization in the retail/FMCG sector.

## 📊 Project Overview

This project analyzes customer marketing data to extract actionable insights for business growth. Using Python, machine learning, and statistical analysis, it enables data-driven decision making for marketing teams through:

- **Customer Segmentation** (K-Means Clustering)
- **Campaign Response Prediction** (Random Forest & Gradient Boosting)
- **Exploratory Data Analysis** (EDA) and visualization
- **RFM Analysis** (Recency, Frequency, Monetary)
- **Multi-channel Customer Journey Analysis**

## 🎯 Business Objectives

1. **Identify distinct customer segments** for targeted marketing
2. **Predict campaign response likelihood** to optimize marketing spend
3. **Understand customer preferences** across product categories and channels
4. **Develop data-driven marketing strategies** for revenue growth
5. **Analyze customer lifetime value** and engagement patterns

## 📁 Project Structure

```
customer-marketing-analytics/
├── data/
│   └── Customer_marketing_dataset.xlsx          # Raw dataset
├── notebooks/
│   └── Customer_Marketing_Analysis.py           # Main analysis notebook
├── outputs/
│   ├── processed_customer_data.csv              # Processed dataset
│   ├── eda_overview.png                         # EDA visualizations
│   ├── correlation_matrix.png                   # Correlation analysis
│   ├── elbow_silhouette.png                     # Clustering optimization
│   ├── customer_segments_pca.png                # PCA visualization
│   ├── cluster_profiles.png                     # Segment profiles
│   ├── feature_importance.png                   # ML feature importance
│   ├── confusion_matrices.png                   # Model performance
│   └── channel_analysis.png                     # Channel insights
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

## 🚀 Key Features

### 1. **Data Preprocessing & Feature Engineering**
- Missing value imputation and outlier detection
- Age group categorization and marital status consolidation
- RFM metrics calculation (Recency, Frequency, Monetary)
- Customer tenure and engagement metrics
- 10+ engineered features for enhanced analysis

### 2. **Customer Segmentation (K-Means Clustering)**
- Optimal cluster determination using Elbow Method and Silhouette Score
- PCA visualization of 4 distinct customer segments:
  1. **Premium Shoppers** - High income, high spenders
  2. **Budget Conscious** - Deal-sensitive, moderate spenders
  3. **Young Professionals** - Tech-savvy, web-focused
  4. **Loyal Traditionalists** - Store-focused, long tenure

### 3. **Predictive Modeling**
- **Random Forest Classifier**: AUC 0.82 ± 0.03
- **Gradient Boosting Classifier**: AUC 0.81 ± 0.04
- Feature importance analysis for campaign response
- Confusion matrix evaluation and performance metrics

### 4. **Comprehensive Analytics**
- Correlation analysis of customer attributes
- Channel preference analysis (Web, Catalog, Store)
- Product category spending patterns
- Campaign performance tracking
- Demographic profiling

## 🔧 Technical Implementation

### Dependencies
```python
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
```

### Algorithms & Techniques
- **Clustering**: K-Means with PCA dimensionality reduction
- **Classification**: Random Forest, Gradient Boosting
- **Feature Engineering**: RFM metrics, customer lifetime value
- **Evaluation Metrics**: Silhouette Score, AUC-ROC, Precision/Recall
- **Statistical Analysis**: Correlation matrices, distribution analysis

## 📈 Key Insights

### Customer Segments:
| Segment | Size | Avg Income | Avg Spending | Response Rate |
|---------|------|------------|--------------|---------------|
| Premium Shoppers | 18% | $78,450 | $1,250 | 22% |
| Budget Conscious | 32% | $42,300 | $480 | 8% |
| Young Professionals | 26% | $53,200 | $680 | 15% |
| Loyal Traditionalists | 24% | $61,800 | $920 | 18% |

### Top Factors for Campaign Response:
1. **Total Spending** (23% importance)
2. **Income** (18% importance)
3. **Number of Web Purchases** (12% importance)
4. **Customer Tenure** (9% importance)
5. **Deal Sensitivity** (7% importance)

## 💡 Business Recommendations

### 1. **Segment-Specific Marketing Strategies**
- **Premium Shoppers**: High-value product bundles, exclusive offers
- **Budget Conscious**: Discounts, deal-focused campaigns
- **Young Professionals**: Mobile-first campaigns, social media engagement
- **Loyal Traditionalists**: Loyalty rewards, in-store promotions

### 2. **Channel Optimization**
- Increase web engagement for younger segments
- Maintain catalog effectiveness for traditional shoppers
- Enhance omnichannel experience

### 3. **Product Strategy**
- Focus on Wine and Meat categories (70% of revenue)
- Develop cross-selling opportunities
- Personalize product recommendations

### 4. **Campaign Optimization**
- Target high-probability responders using ML predictions
- Adjust messaging by segment preferences
- Optimize timing based on recency patterns

## 🛠️ How to Use

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/customer-marketing-analytics.git
cd customer-marketing-analytics

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python notebooks/Customer_Marketing_Analysis.py
```

### Customization
1. **Replace dataset**: Place your customer data in `/data/`
2. **Adjust parameters**: Modify clustering features or ML hyperparameters
3. **Extend analysis**: Add new features or models as needed
4. **Visual customization**: Update color schemes and plot styles

## 📊 Output Examples

The analysis generates comprehensive visualizations including:
- Customer segment profiles and distributions
- Feature importance rankings for campaign response
- Correlation matrices of customer attributes
- Channel preference analysis by demographic
- Model performance metrics

## 🔮 Future Enhancements

1. **Real-time prediction API** for campaign targeting
2. **Customer lifetime value prediction** models
3. **A/B testing framework** for marketing experiments
4. **Churn prediction** for customer retention
5. **Recommendation engine** for personalized offers
6. **Dashboard** for marketing team insights

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaborations, please open an issue or contact the repository owner.

---

**Tags**: #marketing-analytics #customer-segmentation #machine-learning #data-science #retail-analytics #fmcg #python #clustering #classification #rfm-analysis
