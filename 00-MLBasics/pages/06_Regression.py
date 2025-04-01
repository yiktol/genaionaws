# House Price Prediction with Machine Learning Regression - Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set page config
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠",
    layout="wide"
)

# Function to generate synthetic house price data
@st.cache_data
def generate_house_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    sqft = np.random.normal(1500, 500, n_samples)  # Square footage
    bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
    bathrooms = np.random.uniform(1, 4, n_samples)  # Number of bathrooms
    lot_size = np.random.normal(9000, 3000, n_samples)  # Lot size in sqft
    age = np.random.gamma(shape=2, scale=15, size=n_samples)  # Age of house
    garage = np.random.randint(0, 4, n_samples)  # Garage spots
    has_pool = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Pool (yes/no)
    
    # Generate neighborhoods with different price levels
    neighborhoods = np.random.choice(['Rural', 'Suburb', 'Urban'], n_samples, p=[0.3, 0.5, 0.2])
    neighborhood_impact = pd.Series(neighborhoods).map({'Rural': -20000, 'Suburb': 0, 'Urban': 40000}).values
    
    # Generate school districts with different price levels
    school_ratings = np.random.randint(1, 11, n_samples)  # School rating (1-10)
    
    # Create DataFrame
    data = pd.DataFrame({
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'LotSize': lot_size,
        'Age': age,
        'GarageSpaces': garage,
        'HasPool': has_pool,
        'Neighborhood': neighborhoods,
        'SchoolRating': school_ratings
    })
    
    # Calculate price as a function of features with some noise
    # Base price
    price = 80000 + 100 * data['SquareFeet'] + 15000 * data['Bedrooms'] + \
            25000 * data['Bathrooms'] - 300 * data['Age'] + \
            0.1 * data['LotSize'] + 10000 * data['GarageSpaces'] + \
            40000 * data['HasPool'] + 8000 * data['SchoolRating'] + \
            neighborhood_impact
            
    # Add some noise to make the relationship less perfect
    price = price + np.random.normal(0, 25000, n_samples)
    
    # Make sure prices are positive
    price = np.maximum(50000, price)
    
    data['Price'] = price
    
    return data

# Function to train models and return the best one
@st.cache_resource
def train_models(data):
    # Convert categorical features to dummy variables
    data_encoded = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)
    
    # Split features and target
    X = data_encoded.drop('Price', axis=1)
    y = data_encoded['Price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    # Sort models by R² score
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['r2'], reverse=True)}
    
    # Return results, feature names, and scaler
    return results, X_train.columns.tolist(), scaler, X_test

# Generate data
housing_data = generate_house_data()

# Train models
model_results, feature_names, scaler, X_test = train_models(housing_data)

# Main application
st.title("🏠 House Price Prediction")
st.write("""
This application demonstrates regression in machine learning by predicting
house prices based on various features such as size, bedrooms, and location.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Predict House Price", "Model Performance", "Data Exploration"])

with tab1:
    st.header("House Price Calculator")
    
    st.write("Enter the details of the house to get an estimated price.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        square_feet = st.number_input("Square Footage", min_value=500, max_value=5000, value=1500)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=7.0, value=2.0, step=0.5)
    
    with col2:
        lot_size = st.number_input("Lot Size (sq ft)", min_value=1000, max_value=30000, value=8000)
        age = st.number_input("House Age (years)", min_value=0, max_value=100, value=15)
        garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=5, value=2)
    
    with col3:
        has_pool = st.selectbox("Swimming Pool", ["No", "Yes"])
        neighborhood = st.selectbox("Neighborhood", ["Rural", "Suburb", "Urban"])
        school_rating = st.slider("School Rating (1-10)", min_value=1, max_value=10, value=7)
    
    # Convert inputs to feature vector
    has_pool_binary = 1 if has_pool == "Yes" else 0
    
    # Create input dataframe for prediction
    input_data = pd.DataFrame({
        'SquareFeet': [square_feet],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'LotSize': [lot_size],
        'Age': [age],
        'GarageSpaces': [garage_spaces],
        'HasPool': [has_pool_binary],
        'SchoolRating': [school_rating],
        'Neighborhood': [neighborhood]
    })
    
    # Create dummy variables for neighborhood (matching training data format)
    input_encoded = pd.get_dummies(input_data, columns=['Neighborhood'], drop_first=True)
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_names]
    
    # Scale the input data
    input_scaled = scaler.transform(input_encoded)
    
    # Choose model
    selected_model = st.selectbox("Select Model for Prediction", list(model_results.keys()))
    
    if st.button("Predict Price", type="primary"):
        model = model_results[selected_model]['model']
        prediction = model.predict(input_scaled)[0]
        
        st.subheader("Estimated House Price")
        st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
        
        # Feature importance (for tree-based models)
        if selected_model in ['Random Forest', 'Gradient Boosting']:
            st.subheader("Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance[:10], ax=ax)
            ax.set_title(f'Top 10 Feature Importance - {selected_model}')
            st.pyplot(fig)
            
            st.write("These features have the biggest impact on the predicted house price.")
        
        # Coefficients (for linear models)
        elif selected_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            st.subheader("Feature Coefficients")
            
            coefficients = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            coefficients['color'] = coefficients['Coefficient'] > 0
            sns.barplot(x='Coefficient', y='Feature', data=coefficients[:10], palette=['#ff9999', '#66b3ff'], hue='color', dodge=False, ax=ax)
            ax.set_title(f'Top 10 Feature Coefficients - {selected_model}')
            ax.get_legend().remove()
            st.pyplot(fig)
            
            st.write("These coefficients show how much the price changes for each unit increase in the feature.")

with tab2:
    st.header("Model Performance Comparison")
    
    # Convert results to DataFrame for comparison
    model_comparison = pd.DataFrame({
        'Model': list(model_results.keys()),
        'R² Score': [results['r2'] for results in model_results.values()],
        'RMSE': [results['rmse'] for results in model_results.values()],
        'MAE': [results['mae'] for results in model_results.values()]
    })
    
    st.write("### Performance Metrics")
    st.dataframe(model_comparison.style.highlight_max(subset=['R² Score']).highlight_min(subset=['RMSE', 'MAE']))
    
    st.write("### Metrics Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='R² Score', data=model_comparison, ax=ax)
        ax.set_title('R² Score by Model (higher is better)')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='RMSE', data=model_comparison, ax=ax)
        ax.set_title('Root Mean Squared Error by Model (lower is better)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Select a model to visualize predictions
    model_for_viz = st.selectbox("Select Model to Visualize Predictions", list(model_results.keys()))
    
    st.write(f"### Actual vs Predicted Prices - {model_for_viz}")
    
    # Get predictions for selected model
    actual = model_results[model_for_viz]['actual']
    predicted = model_results[model_for_viz]['predictions']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted House Prices - {model_for_viz}')
    st.pyplot(fig)
    
    # Error distribution
    st.write(f"### Prediction Error Distribution - {model_for_viz}")
    errors = predicted - actual
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, ax=ax)
    plt.xlabel('Prediction Error')
    plt.title(f'Distribution of Prediction Errors - {model_for_viz}')
    plt.axvline(x=0, color='r', linestyle='--')
    st.pyplot(fig)
    
    st.info("""
    **Understanding the metrics:**
    - **R² Score**: Proportion of variance explained by the model (higher is better, max=1)
    - **RMSE**: Root Mean Squared Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    
    The ideal model would have high R² and low RMSE/MAE.
    """)

with tab3:
    st.header("Dataset Exploration")
    
    # Show dataset sample
    st.subheader("Sample Data (5 rows)")
    st.dataframe(housing_data.head())
    
    # Dataset statistics
    st.subheader("Dataset Summary Statistics")
    st.dataframe(housing_data.describe())
    
    # Distribution of house prices
    st.subheader("House Price Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(housing_data['Price'], kde=True, ax=ax)
    plt.xlabel('Price ($)')
    plt.title('Distribution of House Prices')
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Feature Correlations")
    
    # Create correlation matrix
    numeric_data = housing_data.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr()
    

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', square=True, ax=ax)
    plt.title('Correlation Matrix of House Features')
    st.pyplot(fig)
    
    # Feature relationships
    st.subheader("Feature Relationships with Price")
    
    # Select feature to explore
    numeric_cols = housing_data.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('Price')  # Remove price since it's our target
    
    selected_feature = st.selectbox("Select feature to analyze:", numeric_cols)
    
    # Scatter plot of selected feature vs price
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=selected_feature, y='Price', data=housing_data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    plt.title(f'Relationship between {selected_feature} and Price')
    st.pyplot(fig)
    
    # Price by categorical feature
    st.subheader("Price by Neighborhood")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Neighborhood', y='Price', data=housing_data, ax=ax)
    plt.title('House Prices by Neighborhood')
    st.pyplot(fig)
    
    # Pairplot for selected features
    st.subheader("Multi-feature Relationships")
    
    if st.button("Generate Pairplot (may take a moment)"):
        pairplot_features = ['Price', 'SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
        fig = sns.pairplot(housing_data[pairplot_features], height=2.5)
        plt.suptitle('Relationships Between Key Features', y=1.02)
        st.pyplot(fig)

# Add a sidebar with additional information
with st.sidebar:
    st.title("About")
    st.info("""
    **House Price Predictor**
    
    This application demonstrates regression machine learning for predicting house prices. 
    The model is trained on synthetic data to illustrate the core principles of machine 
    learning regression.
    
    **Features used:**
    - Square Footage
    - Number of Bedrooms & Bathrooms
    - Lot Size
    - House Age
    - Garage Spaces
    - Pool Presence
    - Neighborhood Type
    - School Rating
    
    **Models Demonstrated:**
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    """)
    
    st.subheader("How Regression Works")
    st.write("""
    Unlike classification which predicts categories, regression predicts continuous values:
    
    1. The model learns patterns between features and house prices from training data
    2. It creates a mathematical function to map features to prices
    3. For new houses, it uses this function to predict a price based on features
    4. The accuracy is measured by how close predictions are to actual prices
    """)
    
    # Dataset size information
    st.subheader("Dataset Information")
    st.write(f"Total houses in dataset: {len(housing_data)}")
    st.write(f"Average house price: ${housing_data['Price'].mean():,.2f}")
    st.write(f"Price range: ${housing_data['Price'].min():,.2f} to ${housing_data['Price'].max():,.2f}")
