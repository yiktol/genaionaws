# Customer Segmentation with Clustering - Streamlit Application (Fixed)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# Set page config
st.set_page_config(
    page_title="Customer Segmentation with Clustering",
    page_icon="👥",
    layout="wide"
)

# Function to generate synthetic customer data
@st.cache_data
def generate_customer_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate customer demographics
    age = np.random.normal(40, 15, n_samples)
    age = np.clip(age, 18, 85)  # Clip ages to reasonable range
    
    # Income (correlated somewhat with age)
    income_noise = np.random.normal(0, 20000, n_samples)
    income = 20000 + (age - 18) * 1500 + income_noise
    income = np.clip(income, 10000, 200000)
    
    # Spending score - how much they spend relative to income
    spending_score_noise = np.random.normal(0, 20, n_samples)
    spending_score = 50 + spending_score_noise
    
    # Higher income might lead to higher spending, but not linearly
    income_effect = (income - income.mean()) / income.std() * 5
    spending_score += income_effect
    spending_score = np.clip(spending_score, 1, 100)
    
    # Generate purchase frequency - times per month
    purchase_freq_noise = np.random.normal(0, 2, n_samples)
    purchase_freq = 2 + spending_score/20 + purchase_freq_noise
    purchase_freq = np.clip(purchase_freq, 0, 30)
    
    # Generate average purchase value
    avg_purchase_noise = np.random.normal(0, 30, n_samples)
    avg_purchase = 20 + income/5000 + avg_purchase_noise
    avg_purchase = np.clip(avg_purchase, 5, 500)
    
    # Total spent last year
    total_spent = purchase_freq * 12 * avg_purchase
    
    # Time as customer (years)
    time_as_customer = np.random.gamma(shape=2, scale=2, size=n_samples)
    time_as_customer = np.clip(time_as_customer, 0.1, 20)
    
    # Create customer loyalty score
    loyalty_noise = np.random.normal(0, 10, n_samples)
    loyalty_score = time_as_customer * 5 + spending_score/10 + loyalty_noise
    loyalty_score = np.clip(loyalty_score, 1, 100)
    
    # Distance from store (miles)
    distance = np.random.gamma(shape=2, scale=3, size=n_samples)
    
    # Online purchase ratio
    online_ratio_noise = np.random.normal(0, 0.15, n_samples)
    online_ratio = 0.5 + distance/50 - time_as_customer/40 + online_ratio_noise
    online_ratio = np.clip(online_ratio, 0, 1)
    
    # Discount affinity (how much they respond to discounts)
    discount_noise = np.random.normal(0, 15, n_samples)
    discount_affinity = 100 - income/4000 + purchase_freq*2 + discount_noise
    discount_affinity = np.clip(discount_affinity, 1, 100)
    
    # Returns percentage
    returns_noise = np.random.gamma(shape=1, scale=2, size=n_samples)
    returns_percentage = returns_noise + (100 - loyalty_score)/20
    returns_percentage = np.clip(returns_percentage, 0, 30)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Age': age.astype(int),
        'Income': income.astype(int),
        'SpendingScore': spending_score,
        'PurchaseFrequency': purchase_freq,
        'AvgPurchaseValue': avg_purchase,
        'TotalSpent': total_spent,
        'TimeAsCustomer': time_as_customer,
        'LoyaltyScore': loyalty_score,
        'DistanceFromStore': distance,
        'OnlineRatio': online_ratio,  # Changed from OnlinePurchaseRatio to OnlineRatio
        'DiscountAffinity': discount_affinity,
        'ReturnsPercentage': returns_percentage
    })
    
    # Add customer IDs
    data['CustomerID'] = ['CUST_' + str(i).zfill(5) for i in range(1, n_samples + 1)]
    
    return data

# Function to perform clustering
@st.cache_resource
def perform_clustering(data, algorithm='kmeans', n_clusters=4, eps=0.5, min_samples=5):
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform clustering based on selected algorithm
    if algorithm == 'kmeans':
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = cluster_model.fit_predict(scaled_data)
        centers = scaler.inverse_transform(cluster_model.cluster_centers_)
        
        # Calculate silhouette score
        try:
            silhouette = silhouette_score(scaled_data, labels)
        except:
            silhouette = 0
        
    elif algorithm == 'dbscan':
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = cluster_model.fit_predict(scaled_data)
        centers = None  # DBSCAN doesn't have explicit centers
        
        # Calculate silhouette score if more than one cluster
        unique_labels = set(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            silhouette = silhouette_score(scaled_data, labels)
        else:
            silhouette = 0
            
    elif algorithm == 'hierarchical':
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = cluster_model.fit_predict(scaled_data)
        centers = None  # Hierarchical doesn't have explicit centers
        
        # Calculate silhouette score
        try:
            silhouette = silhouette_score(scaled_data, labels)
        except:
            silhouette = 0
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    result = {
        'labels': labels,
        'centers': centers,
        'principal_components': principal_components,
        'model': cluster_model,
        'scaler': scaler,
        'pca': pca,
        'silhouette': silhouette,
        'variance_ratio': pca.explained_variance_ratio_
    }
    
    return result

# Function to assign a new customer to a cluster
def predict_cluster(customer_data, clustering_results, algorithm, original_data):
    # Scale the data using the same scaler
    scaler = clustering_results['scaler']
    scaled_data = scaler.transform(customer_data)
    
    # Predict cluster
    if algorithm == 'kmeans':
        cluster = clustering_results['model'].predict(scaled_data)[0]
    elif algorithm == 'dbscan':
        # For DBSCAN, find the nearest neighbor in the original data
        # and assign the new point to that cluster
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(scaler.transform(original_data))
        neighbor_idx = nn.kneighbors(scaled_data, return_distance=False)[0][0]
        cluster = clustering_results['labels'][neighbor_idx]
    elif algorithm == 'hierarchical':
        # For Hierarchical, find the nearest neighbor in the original data
        # and assign the new point to that cluster
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(scaler.transform(original_data))
        neighbor_idx = nn.kneighbors(scaled_data, return_distance=False)[0][0]
        cluster = clustering_results['labels'][neighbor_idx]
    
    # Project to PCA space for visualization
    pca = clustering_results['pca']
    pca_projection = pca.transform(scaled_data)
    
    return cluster, pca_projection

# Function to generate cluster profile descriptions
def get_cluster_profiles(data, labels, algorithm):
    # Add cluster labels to the data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Get the number of clusters
    if algorithm == 'dbscan':
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
    else:
        n_clusters = len(set(labels))
    
    # Initialize list to store profiles
    profiles = {}
    
    # Generate profile for each cluster
    for i in range(n_clusters):
        # Skip noise points for DBSCAN
        if algorithm == 'dbscan' and i == -1:
            continue
        
        # Get data for this cluster
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        
        # Calculate key statistics
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Income'].mean()
        avg_spending = cluster_data['SpendingScore'].mean()
        avg_loyalty = cluster_data['LoyaltyScore'].mean()
        avg_purchase_freq = cluster_data['PurchaseFrequency'].mean()
        avg_online_ratio = cluster_data['OnlineRatio'].mean()  # Changed from OnlinePurchaseRatio to OnlineRatio
        avg_discount = cluster_data['DiscountAffinity'].mean()
        
        # Determine key characteristics
        # Age
        if avg_age < 30:
            age_desc = "Young"
        elif avg_age < 50:
            age_desc = "Middle-aged"
        else:
            age_desc = "Senior"
            
        # Income
        if avg_income < 40000:
            income_desc = "Low income"
        elif avg_income < 100000:
            income_desc = "Middle income"
        else:
            income_desc = "High income"
            
        # Spending
        if avg_spending < 40:
            spending_desc = "Conservative spenders"
        elif avg_spending < 70:
            spending_desc = "Moderate spenders"
        else:
            spending_desc = "Big spenders"
            
        # Loyalty
        if avg_loyalty < 40:
            loyalty_desc = "Low loyalty"
        elif avg_loyalty < 70:
            loyalty_desc = "Moderate loyalty"
        else:
            loyalty_desc = "High loyalty"
            
        # Purchase frequency
        if avg_purchase_freq < 2:
            frequency_desc = "Infrequent shoppers"
        elif avg_purchase_freq < 6:
            frequency_desc = "Regular shoppers"
        else:
            frequency_desc = "Frequent shoppers"
            
        # Online vs in-store
        if avg_online_ratio < 0.3:
            channel_desc = "Primarily in-store"
        elif avg_online_ratio < 0.7:
            channel_desc = "Mixed channel"
        else:
            channel_desc = "Primarily online"
            
        # Price sensitivity
        if avg_discount < 40:
            discount_desc = "Not price sensitive"
        elif avg_discount < 70:
            discount_desc = "Moderately price sensitive"
        else:
            discount_desc = "Highly price sensitive"
            
        # Create descriptive name
        name_parts = [age_desc, income_desc]
        name = " ".join(name_parts)
        
        # Create marketing strategy
        if avg_loyalty > 70 and avg_spending > 60:
            strategy = "VIP Program: Offer exclusive products and personalized services."
        elif avg_loyalty > 50 and avg_online_ratio > 0.6:
            strategy = "Digital Engagement: Focus on mobile app features and online exclusives."
        elif avg_discount > 60:
            strategy = "Promotional Strategy: Regular discounts and flash sales."
        elif avg_age < 35 and avg_online_ratio > 0.5:
            strategy = "Social Media: Influencer partnerships and Instagram/TikTok campaigns."
        elif avg_age > 50 and avg_online_ratio < 0.4:
            strategy = "Traditional Marketing: Print catalogs and in-store events."
        else:
            strategy = "Balanced Approach: Mix of online and offline engagement with targeted promotions."
            
        # Store the profile
        profiles[i] = {
            'name': name,
            'size': len(cluster_data),
            'age': avg_age,
            'income': avg_income,
            'spending': avg_spending,
            'loyalty': avg_loyalty,
            'purchase_freq': avg_purchase_freq,
            'online_ratio': avg_online_ratio,
            'discount_affinity': avg_discount,
            'strategy': strategy,
            'description': f"{name} {spending_desc}, {frequency_desc}, {channel_desc}, {loyalty_desc}, {discount_desc}"
        }
    
    return profiles

# Generate data
original_data = generate_customer_data(1000)

# Main app
st.title("👥 Customer Segmentation with Clustering")
st.write("""
This application demonstrates how machine learning clustering algorithms can be used to segment customers 
based on their characteristics and behaviors. Explore the clusters, visualize patterns, and see how new 
customers would be classified.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Cluster Analysis", "Customer Assignment", "Data Exploration", "About Clustering"])

with tab1:
    st.header("Customer Segmentation")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Clustering Parameters")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["kmeans", "dbscan", "hierarchical"],
            format_func=lambda x: {
                "kmeans": "K-Means",
                "dbscan": "DBSCAN",
                "hierarchical": "Hierarchical"
            }[x]
        )
        
        # Parameters based on selected algorithm
        if algorithm == "kmeans" or algorithm == "hierarchical":
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
            eps = 0.5  # Default value for DBSCAN
            min_samples = 5  # Default value for DBSCAN
        else:  # DBSCAN
            eps = st.slider("Epsilon (Neighborhood Size)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples = st.slider("Minimum Samples", min_value=2, max_value=20, value=5)
            n_clusters = 4  # Default value for KMeans
        
        # Feature selection
        st.subheader("Select Features")
        all_features = original_data.columns.tolist()
        all_features.remove('CustomerID')  # Remove ID from features
        
        default_features = ['Age', 'Income', 'SpendingScore', 'LoyaltyScore', 'PurchaseFrequency', 'OnlineRatio','DiscountAffinity']
        selected_features = st.multiselect(
            "Features to use for clustering",
            options=all_features,
            default=default_features
        )
        
        if not selected_features:
            st.warning("Please select at least one feature.")
            selected_features = default_features
        
        # Run clustering
        data_for_clustering = original_data[selected_features]
        clustering_results = perform_clustering(data_for_clustering, algorithm, n_clusters, eps, min_samples)
        
        # Show silhouette score
        st.metric("Silhouette Score", f"{clustering_results['silhouette']:.3f}")
        st.caption("Higher is better (range: -1 to 1)")
        
        # Show explained variance
        variance = clustering_results['variance_ratio']
        st.metric("PCA Explained Variance", f"{sum(variance):.1%}")
        st.caption(f"PC1: {variance[0]:.1%}, PC2: {variance[1]:.1%}")
    
    with col2:
        # Show the clusters visualized using PCA
        st.subheader("Customer Clusters Visualization")
        
        # Get PCA components and labels
        pca_result = clustering_results['principal_components']
        labels = clustering_results['labels']
        
        # Create a DataFrame for Plotly
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # If using DBSCAN, rename cluster -1 to "Noise"
        if algorithm == 'dbscan':
            pca_df['Cluster'] = pca_df['Cluster'].replace('-1', 'Noise')
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Cluster',
            title='Customer Segments (PCA Projection)',
            opacity=0.7,
            height=500
        )
        
        # Show cluster centers if available
        if clustering_results['centers'] is not None:
            centers_pca = clustering_results['pca'].transform(
                clustering_results['scaler'].transform(clustering_results['centers'])
            )
            
            centers_df = pd.DataFrame({
                'PC1': centers_pca[:, 0],
                'PC2': centers_pca[:, 1]
            })
            
            # Add centers to the plot
            fig.add_trace(
                go.Scatter(
                    x=centers_df['PC1'], 
                    y=centers_df['PC2'],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='black',
                        line=dict(width=1)
                    ),
                    name='Cluster Centers'
                )
            )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
    
    # Get cluster profiles
    cluster_profiles = get_cluster_profiles(data_for_clustering, clustering_results['labels'], algorithm)
    
    # Display cluster profiles
    st.header("Cluster Profiles")
    
    profile_cols = st.columns(min(len(cluster_profiles), 4))
    
    for i, (cluster_id, profile) in enumerate(cluster_profiles.items()):
        with profile_cols[i % len(profile_cols)]:
            st.subheader(f"Cluster {cluster_id}: {profile['name']}")
            
            # Display cluster size
            st.caption(f"{profile['size']} customers ({profile['size'] / len(original_data):.1%} of total)")
            
            # Show key metrics as gauges
            col1, col2 = st.columns(2)
            
            with col1:
                # Create 3 small gauge charts
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = profile['spending'],
                    title = {'text': "Spending"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                    },
                    domain = {'x': [0, 1], 'y': [0, 0.33]}
                ))
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = profile['loyalty'],
                    title = {'text': "Loyalty"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green"},
                    },
                    domain = {'x': [0, 1], 'y': [0.33, 0.66]}
                ))
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = profile['discount_affinity'],
                    title = {'text': "Price Sensitivity"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "firebrick"},
                    },
                    domain = {'x': [0, 1], 'y': [0.66, 1]}
                ))
                
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display profile description
                st.markdown(f"""
                **Age:** {profile['age']:.1f} years
                
                **Income:** ${profile['income']:,.0f}
                
                **Shopping Frequency:** {profile['purchase_freq']:.1f} times/month
                
                **Online Shopping:** {profile['online_ratio']*100:.1f}%
                """)
            
            # Show marketing strategy
            st.markdown(f"**Recommended Strategy:** {profile['strategy']}")

with tab2:
    st.header("Assign New Customer to Cluster")
    
    st.write("""
    Enter customer information below to see which segment they would belong to.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=85, value=35)
        income = st.slider("Annual Income ($)", min_value=10000, max_value=200000, value=60000, step=1000)
        spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        purchase_freq = st.slider("Purchases per Month", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    
    with col2:
        loyalty_score = st.slider("Loyalty Score (1-100)", min_value=1, max_value=100, value=50)
        online_ratio = st.slider("Online Purchase Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        discount_affinity = st.slider("Price Sensitivity (1-100)", min_value=1, max_value=100, value=50)
        time_as_customer = st.slider("Years as Customer", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    
    # Create additional derived features
    avg_purchase = st.slider("Average Purchase Value ($)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)
    total_spent = purchase_freq * 12 * avg_purchase
    distance = st.slider("Distance from Store (miles)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    returns_percentage = st.slider("Returns Percentage", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    
    # Create customer data
    new_customer_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'SpendingScore': [spending_score],
        'PurchaseFrequency': [purchase_freq],
        'AvgPurchaseValue': [avg_purchase],
        'TotalSpent': [total_spent],
        'TimeAsCustomer': [time_as_customer],
        'LoyaltyScore': [loyalty_score],
        'DistanceFromStore': [distance],
        'OnlineRatio': [online_ratio],  # Changed from OnlinePurchaseRatio to OnlineRatio
        'DiscountAffinity': [discount_affinity],
        'ReturnsPercentage': [returns_percentage]
    })
    
    # Use only selected features
    new_customer_selected_features = new_customer_data[selected_features]
    
    # Predict cluster
    if st.button("Predict Customer Segment", type="primary"):
        cluster, pca_projection = predict_cluster(
            new_customer_selected_features, 
            clustering_results, 
            algorithm,
            data_for_clustering
        )
        
        # Display the result
        st.subheader("Customer Segment Prediction")
        
        # Handle DBSCAN noise points
        if algorithm == 'dbscan' and cluster == -1:
            st.warning("This customer is classified as an **outlier** (doesn't fit well into any segment).")
            segment_name = "Outlier"
        else:
            # Get cluster profile
            profile = cluster_profiles.get(cluster, {'name': 'Unknown', 'description': '', 'strategy': ''})
            segment_name = profile['name']
            
            # Show the cluster information
            st.success(f"This customer belongs to **Cluster {cluster}: {segment_name}**")
            
            # Show description and strategy
            st.markdown(f"""
            **Segment Description:** {profile.get('description', 'No description available')}
            
            **Recommended Marketing Strategy:** {profile.get('strategy', 'No strategy available')}
            """)
        
        # Show the customer in the PCA space
        st.subheader("Customer Position in Segment Space")
        
        # Get PCA components and labels for visualization
        pca_result = clustering_results['principal_components']
        labels = clustering_results['labels']
        
        # Create a DataFrame for the plot
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': labels.astype(str)
        })
        
        # If using DBSCAN, rename cluster -1 to "Noise"
        if algorithm == 'dbscan':
            pca_df['Cluster'] = pca_df['Cluster'].replace('-1', 'Noise')
        
        # Create scatter plot
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Cluster',
            opacity=0.2,
            height=500
        )
        
        # Add new customer point
        fig.add_trace(
            go.Scatter(
                x=[pca_projection[0, 0]],
                y=[pca_projection[0, 1]],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color='yellow',
                    line=dict(color='black', width=2)
                ),
                name='New Customer'
            )
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Customer Data Exploration")
    
    # Show data sample
    st.subheader("Sample Customer Data")
    st.dataframe(original_data.head(10))
    
    # Feature correlations
    st.subheader("Feature Correlations")
    
    # Calculate correlation matrix
    corr_matrix = original_data.drop(columns=['CustomerID']).corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select feature to visualize
    feature_to_plot = st.selectbox(
        "Select Feature to Visualize",
        options=all_features
    )
    
    # Create distribution plot
    fig = px.histogram(
        original_data, x=feature_to_plot,
        marginal="box",
        title=f"Distribution of {feature_to_plot}",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.subheader("Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis Feature", options=all_features, key="x_feature")
    
    with col2:
        y_feature = st.selectbox("Y-axis Feature", options=all_features, key="y_feature", index=1)
    
    # Create scatter plot
    fig = px.scatter(
        original_data, x=x_feature, y=y_feature,
        opacity=0.6,
        title=f"{x_feature} vs {y_feature}",
        height=500,
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("About Clustering")
    
    st.markdown("""
    ## What is Clustering?
    
    Clustering is an **unsupervised machine learning** technique that groups similar data points together based on their features, without requiring labeled data. The goal is to find natural groupings in data.
    
    ### Key Characteristics:
    - **Unsupervised learning**: No predefined labels or categories
    - **Similarity-based**: Groups are formed based on similar characteristics
    - **Exploratory analysis**: Helps discover hidden patterns and structures
    
    ### Common Algorithms:
    
    #### K-Means
    - Partitions data into K clusters
    - Each data point belongs to the cluster with the nearest mean
    - Good for well-separated, spherical clusters
    
    #### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    - Groups points that are closely packed together
    - Can identify outliers and find clusters of arbitrary shapes
    - Doesn't require specifying the number of clusters
    
    #### Hierarchical Clustering
    - Builds a tree of clusters (dendrogram)
    - Can be agglomerative (bottom-up) or divisive (top-down)
    - Provides multiple levels of clustering granularity
    
    ### Applications in Customer Segmentation:
    - **Marketing Targeting**: Tailor marketing strategies to specific customer groups
    - **Product Recommendations**: Suggest products based on segment preferences
    - **Service Personalization**: Customize services for different customer types
    - **Retention Strategies**: Develop specific programs for at-risk segments
    """)
    
    st.image("https://miro.medium.com/max/1400/1*ET8kCcPpr893vNZFs8j4xg.png", caption="Visualization of different clustering algorithms")
    

    st.markdown("""
    ### Evaluating Clustering Quality
    
    Since clustering is unsupervised, evaluation is more challenging than supervised methods:
    
    - **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (-1 to 1, higher is better)
    - **Inertia/SSE**: Sum of squared distances to the nearest cluster center (lower is better)
    - **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
    - **Business Interpretability**: How meaningful and actionable are the discovered segments
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.title("About Customer Segmentation")
    
    st.info("""
    **Customer Segmentation**
    
    This application demonstrates clustering algorithms to segment customers based on their 
    characteristics and behaviors. Use this tool to identify distinct customer groups and 
    develop targeted marketing strategies.
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. **Choose an algorithm**: Select K-Means, DBSCAN, or Hierarchical clustering
    2. **Adjust parameters**: Set number of clusters, epsilon, etc.
    3. **Select features**: Choose which customer attributes to consider
    4. **Explore clusters**: Examine the distinctive characteristics of each segment
    5. **Assign new customers**: See which segment a new customer would belong to
    """)
    
    st.subheader("Dataset Information")
    st.markdown(f"""
    - **Total Customers**: {len(original_data)}
    - **Features**: {len(all_features)}
    - **Synthetic Data**: This demonstration uses generated data that mimics real customer behavior patterns
    """)
    
    # Add download option for the dataset
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(original_data)
    st.download_button(
        label="Download Customer Dataset",
        data=csv,
        file_name='customer_segmentation_data.csv',
        mime='text/csv',
    )
    
    # Add elbow method for K-Means
    if algorithm == "kmeans":
        st.subheader("Finding Optimal K")
        st.write("Use the Elbow Method to find the optimal number of clusters:")
        
        if st.button("Calculate Elbow Curve"):
            # Calculate inertia for different K values
            inertia = []
            k_range = range(1, 11)
            
            with st.spinner('Calculating Elbow Curve...'):
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(clustering_results['scaler'].transform(data_for_clustering))
                    inertia.append(kmeans.inertia_)
            
            # Plot elbow curve
            fig = px.line(
                x=list(k_range), y=inertia,
                markers=True,
                labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                title='Elbow Method for Optimal K'
            )
            fig.add_annotation(
                text="Optimal K is where the curve bends the most",
                x=4, y=inertia[3],
                showarrow=True,
                arrowhead=1
            )
            st.plotly_chart(fig, use_container_width=True)
