# Machine Learning Terminology Explorer - Streamlit Application (California Housing Dataset)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

# Configure the page
st.set_page_config(
    page_title="ML Terminology Explorer",
    page_icon="📊",
    layout="wide"
)

# Create a custom dataset that makes terminology clear
@st.cache_data
def create_custom_dataset(n_samples=100):
    """Create a simple dataset for house prices with clear features."""
    np.random.seed(42)
    
    # Features
    square_feet = np.random.randint(1000, 4000, size=n_samples)  # Square footage of house
    num_bedrooms = np.random.randint(1, 6, size=n_samples)      # Number of bedrooms
    num_bathrooms = np.random.randint(1, 4, size=n_samples)     # Number of bathrooms
    age_of_house = np.random.randint(0, 50, size=n_samples)     # Age of house in years
    
    # Target variable (house price) with some noise
    price = (
        150000 +                               # Base price
        100 * square_feet +                   # $100 per square foot
        15000 * num_bedrooms +                # $15,000 per bedroom
        20000 * num_bathrooms +               # $20,000 per bathroom
        -1000 * age_of_house +                # -$1,000 per year of age
        np.random.normal(0, 20000, n_samples) # Random noise
    )
    
    # Create a DataFrame
    df = pd.DataFrame({
        'SquareFeet': square_feet,
        'Bedrooms': num_bedrooms,
        'Bathrooms': num_bathrooms,
        'HouseAge': age_of_house,
        'Price': price
    })
    
    return df

# Load built-in datasets
@st.cache_data
def load_datasets():
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['target_name'] = [iris.target_names[t] for t in iris.target]
    
    # Load California Housing dataset
    california = fetch_california_housing()
    california_df = pd.DataFrame(california.data, columns=california.feature_names)
    california_df['target'] = california.target
    
    # Load Diabetes dataset
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    
    # Load Wine dataset
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df['target_name'] = [wine.target_names[t] for t in wine.target]
    
    # Create custom dataset
    custom_df = create_custom_dataset()
    
    return {
        'House Price (Custom)': custom_df,
        'California Housing': california_df,
        'Iris Classification': iris_df,
        'Wine Classification': wine_df,
        'Diabetes Regression': diabetes_df
    }

# Dictionary of ML terminology with explanations
ml_terminology = {
    'Dataset': {
        'definition': 'A collection of data used for machine learning.',
        'example': 'The California Housing dataset contains information about housing districts in California.',
        'visual_type': 'table'
    },
    'Features (Inputs/Predictors/Independent Variables)': {
        'definition': 'Individual measurable properties or characteristics of the phenomena being observed.',
        'example': 'In a housing dataset, features might include median income, house age, average rooms, etc.',
        'visual_type': 'column_highlight'
    },
    'Target (Label/Output/Dependent Variable)': {
        'definition': 'The variable you want to predict or classify.',
        'example': 'In the California Housing dataset, the target is the median house value.',
        'visual_type': 'column_highlight'
    },
    'Observation (Instance/Sample/Example)': {
        'definition': 'A single data point or record in the dataset.',
        'example': 'In the California Housing dataset, each row represents a housing district with its features.',
        'visual_type': 'row_highlight'
    },
    'Feature Matrix (X)': {
        'definition': 'The entire set of input features, typically represented as a 2D matrix where rows are observations and columns are features.',
        'example': 'All columns except "target" in the California Housing dataset form the feature matrix.',
        'visual_type': 'matrix'
    },
    'Target Vector (y)': {
        'definition': 'The collection of all target values, typically represented as a 1D vector.',
        'example': 'The "target" column in the California Housing dataset is the target vector.',
        'visual_type': 'vector'
    },
    'Training Set': {
        'definition': 'A subset of the data used to train the model.',
        'example': '70-80% of the housing data used for the model to learn patterns.',
        'visual_type': 'split'
    },
    'Testing Set': {
        'definition': 'A subset of the data used to evaluate the model\'s performance.',
        'example': '20-30% of the housing data kept separate to test the model.',
        'visual_type': 'split'
    },
    'Validation Set': {
        'definition': 'A subset of the training data used to tune hyperparameters and prevent overfitting.',
        'example': 'A portion of the training data used to validate model performance during training.',
        'visual_type': 'split'
    },
    'Classification': {
        'definition': 'Predicting categorical class labels or discrete outcomes.',
        'example': 'Predicting whether an email is spam or not spam.',
        'visual_type': 'classification'
    },
    'Regression': {
        'definition': 'Predicting continuous numerical values.',
        'example': 'Predicting house prices based on features.',
        'visual_type': 'regression'
    },
    'Model': {
        'definition': 'An algorithm or mathematical construct that learns patterns from data to make predictions.',
        'example': 'Linear regression, random forest, or neural networks are examples of models.',
        'visual_type': 'model'
    },
    'Prediction': {
        'definition': 'The output value(s) produced by a trained model when given new input data.',
        'example': 'The estimated median house value based on district features.',
        'visual_type': 'prediction'
    },
    'Accuracy': {
        'definition': 'The proportion of correct predictions made by the model.',
        'example': 'If a model correctly classifies 90 out of 100 emails as spam or not spam, the accuracy is 90%.',
        'visual_type': 'metrics'
    },
    'Overfitting': {
        'definition': 'When a model learns the training data too well, including the noise, leading to poor performance on new data.',
        'example': 'A model that perfectly predicts house values in the training data but fails on new districts.',
        'visual_type': 'overfitting'
    },
    'Underfitting': {
        'definition': 'When a model is too simple to capture the underlying pattern in the data.',
        'example': 'Using a linear model to predict house values when the relationship is non-linear.',
        'visual_type': 'underfitting'
    }
}

# Load datasets
datasets = load_datasets()

# Title and introduction
st.title("📊 Machine Learning Terminology Explorer")
st.markdown("""
This interactive app explains core machine learning terminology through visualizations and examples. 
Choose a term from the dropdown menu to see its definition, example, and visual representation.
""")

# Sidebar with term selection
st.sidebar.title("ML Terminology")
selected_term = st.sidebar.selectbox("Select a term to explore:", list(ml_terminology.keys()))
st.sidebar.markdown("---")

# Dataset selection
selected_dataset = st.sidebar.selectbox("Select a dataset to visualize:", list(datasets.keys()))
current_df = datasets[selected_dataset]

# Show sample data in sidebar
st.sidebar.markdown("### Sample Data")
st.sidebar.dataframe(current_df.head(3))

# Main content
st.header(selected_term)

# Display definition and example
term_info = ml_terminology[selected_term]
st.markdown(f"""
**Definition:** {term_info['definition']}

**Example:** {term_info['example']}
""")

# Visual explanations based on term type
if term_info['visual_type'] == 'table':
    st.subheader("Visual Explanation: Dataset")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        A **dataset** is a collection of observations (rows) with their features and target values.
        
        Key components of a dataset:
        - **Features**: The input variables (columns)
        - **Target**: The output variable to predict
        - **Observations**: Individual data points (rows)
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(current_df.columns),
                fill_color='#4CAF50',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[current_df[col] for col in current_df.columns],
                fill_color='lavender',
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Complete Dataset',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif term_info['visual_type'] == 'column_highlight':
    st.subheader(f"Visual Explanation: {'Features' if 'Features' in selected_term else 'Target'}")
    
    if 'Features' in selected_term:
        highlight_cols = current_df.columns[:-1]  # All columns except the last (assuming last column is target)
        non_highlight_cols = current_df.columns[-1:]
        title = "Features are the input variables (highlighted columns)"
    else:  # Target
        non_highlight_cols = current_df.columns[:-1]
        highlight_cols = current_df.columns[-1:]
        title = "Target is the output variable to predict (highlighted column)"
    
    # Create a DataFrame with highlighted columns
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(current_df.columns),
            fill_color=['#4CAF50' if col in highlight_cols else '#A9A9A9' for col in current_df.columns],
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[current_df[col] for col in current_df.columns],
            fill_color=['lavender' if col in highlight_cols else '#f0f0f0' for col in current_df.columns],
            align='center'
        )
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif term_info['visual_type'] == 'row_highlight':
    st.subheader("Visual Explanation: Observation")
    
    # Choose a random row to highlight
    highlight_row = np.random.randint(0, len(current_df))
    
    # Display the dataframe with the highlighted row
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(current_df.columns),
            fill_color='#4CAF50',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[current_df[col] for col in current_df.columns],
            fill_color=[['#FFC107' if i == highlight_row else 'lavender' for i in range(len(current_df))] for _ in range(len(current_df.columns))],
            align='center'
        )
    )])
    
    fig.update_layout(
        title='An Observation is a single data point (highlighted row)',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the observation as a "record"
    st.markdown("### Observation as a Record")
    
    observation = current_df.iloc[highlight_row].to_dict()
    
    for feature, value in observation.items():
        if feature == current_df.columns[-1]:
            st.metric(f"Target: {feature}", f"{value:.2f}" if isinstance(value, (float, np.floating)) else f"{value}")
        else:
            st.metric(f"Feature: {feature}", f"{value:.2f}" if isinstance(value, (float, np.floating)) else f"{value}")

elif term_info['visual_type'] == 'matrix':
    st.subheader("Visual Explanation: Feature Matrix (X)")
    
    # Assuming the last column is the target
    X = current_df.iloc[:, :-1]
    y = current_df.iloc[:, -1]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        The **Feature Matrix (X)** contains all input features for all observations.
        
        It's a 2D matrix where:
        - Each **row** is an observation
        - Each **column** is a feature
        
        In mathematical notation, X often represents the feature matrix.
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(X.columns),
                fill_color='#4CAF50',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[X[col] for col in X.columns],
                fill_color='lavender',
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Feature Matrix (X): All input features for all observations',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show matrix shape
    st.markdown(f"**Feature Matrix Shape:** {X.shape[0]} rows (observations) × {X.shape[1]} columns (features)")

elif term_info['visual_type'] == 'vector':
    st.subheader("Visual Explanation: Target Vector (y)")
    
    # Assuming the last column is the target
    y = current_df.iloc[:, -1]
    target_name = current_df.columns[-1]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"""
        The **Target Vector (y)** contains all target values.
        
        It's a 1D vector where:
        - Each element corresponds to an observation's target value
        - In this case, the target is **{target_name}**
        
        In mathematical notation, y often represents the target vector.
        """)
    
    with col2:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[target_name],
                fill_color='#E91E63',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[y],
                fill_color='lavender',
                align='center'
            )
        )])
        
        fig.update_layout(
            title='Target Vector (y): All target values',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show target distribution
    st.subheader("Target Distribution")
    
    if len(y.unique()) < 10:  # Categorical target
        fig = px.histogram(current_df, x=target_name, color=target_name if 'target_name' in current_df.columns else None)
    else:  # Continuous target
        fig = px.histogram(current_df, x=target_name)
    
    st.plotly_chart(fig, use_container_width=True)

elif term_info['visual_type'] == 'split':
    st.subheader("Visual Explanation: Data Splitting")
    
    # Create a sample split
    train_size = 0.7 if 'Training' in selected_term else 0.6
    test_size = 0.3 if 'Testing' in selected_term else 0.2
    val_size = 0.0 if 'Validation' not in selected_term else 0.2
    
    # Generate indices for splitting
    n_samples = len(current_df)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(train_size * n_samples)
    test_start = n_samples - int(test_size * n_samples)
    
    train_indices = indices[:train_end]
    test_indices = indices[test_start:]
    val_indices = indices[train_end:test_start] if val_size > 0 else []
    
    # Create visual representation of split
    split_df = pd.DataFrame({
        'index': range(n_samples),
        'set': ['Training' if i in train_indices else 
                'Testing' if i in test_indices else 
                'Validation' for i in range(n_samples)]
    })
    
    colors = {'Training': '#4CAF50', 'Testing': '#F44336', 'Validation': '#2196F3'}
    
    fig = px.scatter(split_df, x='index', y=[1] * n_samples, 
                   color='set', color_discrete_map=colors,
                   title=f"Data Splitting: {int(train_size*100)}% Training, {int(test_size*100)}% Testing{f', {int(val_size*100)}% Validation' if val_size > 0 else ''}",
                   labels={'index': 'Data points', 'y': ''})
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=300, showlegend=True)
    fig.update_yaxes(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the purpose
    if 'Training' in selected_term:
        st.markdown("""
        The **Training Set** is used to train the model. This is where the model learns patterns from the data.
        
        Key points:
        - Usually the largest portion (70-80%) of the data
        - The model sees these examples during training
        - The model adjusts its parameters based on this data
        """)
    elif 'Testing' in selected_term:
        st.markdown("""
        The **Testing Set** is used to evaluate the model's performance on unseen data.
        
        Key points:
        - Usually 20-30% of the data
        - The model never sees this data during training
        - Used to estimate how well the model will perform on new, unseen data
        - Helps detect overfitting (when a model performs well on training data but poorly on test data)
        """)
    else:  # Validation
        st.markdown("""
        The **Validation Set** is used during the training process to tune hyperparameters and prevent overfitting.
        
        Key points:
        - A portion of the training data set aside
        - Used to evaluate the model during training, not after
        - Helps in selecting the best model configuration
        - Different from the test set, which is only used after training is complete
        """)

elif term_info['visual_type'] == 'classification':
    st.subheader("Visual Explanation: Classification")
    
    # Use a classification dataset
    if 'target_name' in current_df.columns:
        df = current_df
    else:
        df = datasets['Iris Classification']
    
    # Select two features for visualization
    if 'Iris' in selected_dataset:
        feature1, feature2 = 'sepal length (cm)', 'sepal width (cm)'
        target = 'target_name'
    elif 'Wine' in selected_dataset:
        feature1, feature2 = df.columns[0], df.columns[1]
        target = 'target_name'
    else:
        # Generic features
        feature1, feature2 = df.columns[0], df.columns[1]
        target = df.columns[-1]
    
    # Create scatter plot
    fig = px.scatter(df, x=feature1, y=feature2, color=target,
                   title="Classification: Predicting Categorical Outcomes",
                   labels={feature1: feature1, feature2: feature2, target: "Class"})
    
    # Add decision boundary (simulated for visualization)
    if len(df[target].unique()) <= 3:  # Only if we have a reasonable number of classes
        X = df[[feature1, feature2]].values
        y = df[target].astype('category').cat.codes.values
        
        # Train a simple model to get a decision boundary
        model = RandomForestClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                           np.arange(y_min, y_max, 0.1))
        
        # Predict on the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Add contour lines to show decision boundaries
        fig.add_trace(
            go.Contour(
                x=np.arange(x_min, x_max, 0.1),
                y=np.arange(y_min, y_max, 0.1),
                z=Z,
                showscale=False,
                colorscale='Viridis',
                opacity=0.4,
                line=dict(width=0),
                contours=dict(showlabels=False)
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Classification** is a type of supervised learning where the model predicts categorical outcomes.
    
    Examples:
    - Email spam detection (spam/not spam)
    - Disease diagnosis (positive/negative)
    - Image recognition (cat/dog/bird/etc.)
    
    In the plot above:
    - Each point represents an observation
    - Colors represent different classes
    - The model learns to separate the classes based on features
    """)

elif term_info['visual_type'] == 'regression':
    st.subheader("Visual Explanation: Regression")
    
    # Use a regression dataset
    if 'Price' in current_df.columns or 'California' in selected_dataset:
        df = current_df
        if 'California' in selected_dataset:
            feature = 'MedInc'  # Median Income
            target = 'target'   # Median house value
        else:
            feature = 'SquareFeet'
            target = 'Price'
    else:
        df = datasets['Diabetes Regression']
        feature = df.columns[0]  # First feature
        target = 'target'
    
    # Create scatter plot
    fig = px.scatter(df, x=feature, y=target,
                   title="Regression: Predicting Continuous Values",
                   labels={feature: feature, target: target})
    
    # Add regression line
    X = df[feature].values.reshape(-1, 1)
    y = df[target].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    fig.add_trace(
        go.Scatter(
            x=df[feature],
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=3)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show equation
    slope = model.coef_[0]
    intercept = model.intercept_
    
    st.markdown(f"""
    **Regression** is a type of supervised learning where the model predicts continuous values.
    
    Examples:
    - House price prediction
    - Stock price forecasting
    - Age estimation
    - Sales forecasting
    
    In the plot above:
    - Each point represents an observation
    - The red line shows the predicted relationship between the feature and target
    
    **Regression Line Equation:**
    {target} = {slope:.2f} × {feature} + {intercept:.2f}
    """)

elif term_info['visual_type'] == 'model':
    st.subheader("Visual Explanation: Model")
    
    # Use the California Housing dataset
    if 'California' in selected_dataset:
        df = current_df
        target_col = 'target'
    else:
        df = datasets['California Housing']
        target_col = 'target'
    
    # Split data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Extract coefficients
    coeffs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    
    # Create visualization
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        A **Model** is an algorithm that learns patterns from data to make predictions.
        
        Models can be:
        - Simple (linear regression)
        - Complex (neural networks)
        - Interpretable or black-box
        
        A trained model has learned parameters (coefficients, weights, etc.) 
        that determine how it maps inputs to outputs.
        """)
    
    with col2:
        # Plot coefficients
        fig = px.bar(coeffs, x='Feature', y='Coefficient',
                   title="Linear Regression Model Coefficients",
                   labels={'Coefficient': 'Impact on House Value'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
        
    with col2:
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred, 
                       labels={'x': 'Actual Value', 'y': 'Predicted Value'},
                       title='Model Predictions: Actual vs Predicted')
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Predictions'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif term_info['visual_type'] == 'prediction':
    st.subheader("Visual Explanation: Prediction")
    
    # Use California Housing dataset
    if 'California' in selected_dataset:
        df = current_df
    else:
        df = datasets['California Housing']
    
    # Train a simple model
    X = df.drop('target', axis=1)
    y = df['target']
    
    model = LinearRegression().fit(X, y)
    
    # Create input form with California Housing features
    st.markdown("""
    Prediction is the output produced by a model when given input features.
    Try making a prediction by adjusting the input features below:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        median_income = st.slider("Median Income (tens of thousands)", 
                               float(X['MedInc'].min()), 
                               float(X['MedInc'].max()), 
                               3.5)
        
        house_age = st.slider("Housing Median Age", 
                           float(X['HouseAge'].min()), 
                           float(X['HouseAge'].max()), 
                           28.0)
        
        avg_rooms = st.slider("Average Rooms per Household", 
                           float(X['AveRooms'].min()), 
                           float(X['AveRooms'].max()), 
                           5.0)
        
        avg_bedrooms = st.slider("Average Bedrooms per Household", 
                              float(X['AveBedrms'].min()), 
                              float(X['AveBedrms'].max()), 
                              1.0)
    
    with col2:
        population = st.slider("Block Population", 
                            float(X['Population'].min()), 
                            float(X['Population'].max()), 
                            1500.0)
        
        avg_occupancy = st.slider("Average Household Occupancy", 
                               float(X['AveOccup'].min()), 
                               float(X['AveOccup'].max()), 
                               3.0)
        
        latitude = st.slider("Latitude", 
                          float(X['Latitude'].min()), 
                          float(X['Latitude'].max()), 
                          35.0)
        
        longitude = st.slider("Longitude", 
                           float(X['Longitude'].min()), 
                           float(X['Longitude'].max()), 
                           -120.0)
    

    # Make prediction
    new_district = pd.DataFrame({
        'MedInc': [median_income],
        'HouseAge': [house_age],
        'AveRooms': [avg_rooms],
        'AveBedrms': [avg_bedrooms],
        'Population': [population],
        'AveOccup': [avg_occupancy],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    
    predicted_value = model.predict(new_district)[0]
    
    # Display prediction with nice formatting
    st.markdown("### Model Prediction")
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='margin: 0;'>${predicted_value * 100000:,.2f}</h2>
        <p>Predicted Median House Value</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show how the prediction was made
    st.markdown("### How the Prediction Works")
    
    # Calculate contribution of each feature
    intercept = model.intercept_
    contributions = {}
    
    for i, feature in enumerate(X.columns):
        contribution = model.coef_[i] * new_district[feature].values[0]
        contributions[feature] = contribution
    
    # Create waterfall chart data
    waterfall_labels = list(contributions.keys()) + ['Intercept', 'Predicted Value']
    waterfall_values = list(contributions.values()) + [intercept, 0]
    
    # Calculate the cumulative sum for the measure
    measure = ['relative'] * len(contributions) + ['absolute', 'total']
    
    fig = go.Figure(go.Waterfall(
        name="Prediction Breakdown",
        orientation="v",
        measure=measure,
        x=waterfall_labels,
        y=waterfall_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#4CAF50"}},
        decreasing={"marker": {"color": "#F44336"}},
        totals={"marker": {"color": "#2196F3"}}
    ))
    
    fig.update_layout(
        title="Contribution of Each Feature to the Prediction",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a more readable explanation
    st.markdown("**Explanation of Feature Contributions:**")
    
    # Format the contributions for better readability
    st.markdown(f"""
    The prediction is calculated as:
    - Base value (Intercept): ${intercept * 100000:,.2f}
    """)
    
    for feature, contribution in contributions.items():
        contribution_value = contribution * 100000  # Convert to dollar value
        icon = "📈" if contribution > 0 else "📉"
        st.markdown(f"- {icon} {feature}: ${contribution_value:,.2f}")
    
    st.markdown(f"- **Total: ${predicted_value * 100000:,.2f}**")

elif term_info['visual_type'] == 'metrics':
    st.subheader("Visual Explanation: Model Evaluation Metrics")
    
    # Use a classification dataset for simplicity
    if 'target_name' in current_df.columns:
        df = current_df
    else:
        df = datasets['Iris Classification']
    
    # Split data
    X = df.drop(['target', 'target_name'] if 'target_name' in df.columns else ['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    st.markdown("""
    **Evaluation metrics** measure how well a model performs. Different metrics are used for different types of problems.
    
    Common metrics include:
    - **Accuracy**: Proportion of correct predictions (classification)
    - **Precision**: Proportion of positive identifications that were actually correct
    - **Recall**: Proportion of actual positives that were identified correctly
    - **F1 Score**: Harmonic mean of precision and recall
    - **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values (regression)
    - **R²**: Proportion of variance explained by the model (regression)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
        
        if 'target_name' in df.columns:
            st.write("Model correctly classified:")
            for i, class_name in enumerate(np.unique(df['target_name'])):
                correct = cm[i, i]
                total = np.sum(cm[i, :])
                st.write(f"- {class_name}: {correct}/{total} ({correct/total:.0%})")
    
    with col2:
        # Create confusion matrix plot
        fig = px.imshow(cm,
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=[f'Class {i}' for i in range(len(cm))],
                       y=[f'Class {i}' for i in range(len(cm))],
                       text_auto=True,
                       title="Confusion Matrix",
                       color_continuous_scale="Blues")
                       
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Understanding the Confusion Matrix
    
    The confusion matrix shows:
    - **True Positives (TP)**: Correctly predicted positive cases (diagonal elements)
    - **False Positives (FP)**: Incorrectly predicted as positive (off-diagonal columns)
    - **False Negatives (FN)**: Incorrectly predicted as negative (off-diagonal rows)
    - **True Negatives (TN)**: Correctly predicted negative cases
    
    From these, we can calculate metrics like precision (TP/(TP+FP)) and recall (TP/(TP+FN)).
    """)

elif term_info['visual_type'] == 'overfitting':
    st.subheader("Visual Explanation: Overfitting")
    
    # Generate a synthetic dataset to demonstrate overfitting
    np.random.seed(0)
    n_samples = 30
    x = np.linspace(0, 10, n_samples)
    y = 0.5 * x + np.sin(x) + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    df_overfit = pd.DataFrame({'x': x, 'y': y})
    
    # Function to fit polynomial models of different degrees
    def fit_polynomial(x, y, degree):
        model = np.poly1d(np.polyfit(x, y, degree))
        x_line = np.linspace(min(x), max(x), 100)
        y_line = model(x_line)
        return x_line, y_line, model
    
    # Fit models of different complexity
    x_line_simple, y_line_simple, model_simple = fit_polynomial(x, y, 1)    # Underfitting
    x_line_good, y_line_good, model_good = fit_polynomial(x, y, 3)         # Good fit
    x_line_complex, y_line_complex, model_complex = fit_polynomial(x, y, 15)  # Overfitting
    
    # Calculate train and test errors
    # Split data for demonstration
    x_train, x_test = x[:20], x[20:]
    y_train, y_test = y[:20], y[20:]
    
    models = {
        'Underfit (Linear)': model_simple,
        'Good Fit': model_good,
        'Overfit': model_complex
    }
    
    train_errors = {}
    test_errors = {}
    
    for name, model in models.items():
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        train_errors[name] = train_mse
        test_errors[name] = test_mse
    
    # Visualization
    st.markdown("""
    **Overfitting** occurs when a model learns the training data too well, capturing noise rather than just the underlying pattern.
    
    When a model overfits:
    - It performs very well on training data
    - It performs poorly on new, unseen data
    - It has "memorized" the training examples instead of learning generalizable patterns
    """)
    
    # Create plots
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x_train, y=y_train,
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_test, y=y_test,
        mode='markers',
        name='Testing Data',
        marker=dict(color='green', size=10)
    ))
    
    # Add model lines
    fig.add_trace(go.Scatter(
        x=x_line_simple, y=y_line_simple,
        mode='lines',
        name='Underfitting (Linear)',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_good, y=y_line_good,
        mode='lines',
        name='Good Fit',
        line=dict(color='purple', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_complex, y=y_line_complex,
        mode='lines',
        name='Overfitting',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title='Demonstration of Overfitting',
        xaxis_title='x',
        yaxis_title='y',
        legend_title='Legend',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show errors comparison
    st.subheader("Model Performance Comparison")
    
    # Create DataFrame for errors
    error_df = pd.DataFrame({
        'Model': list(train_errors.keys()),
        'Training Error': list(train_errors.values()),
        'Testing Error': list(test_errors.values())
    })
    
    # Plot errors
    fig = go.Figure(data=[
        go.Bar(name='Training Error', x=error_df['Model'], y=error_df['Training Error']),
        go.Bar(name='Testing Error', x=error_df['Model'], y=error_df['Testing Error'])
    ])
    
    fig.update_layout(
        title='Training vs. Testing Error',
        xaxis_title='Model',
        yaxis_title='Mean Squared Error',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **Key Observations:**
    
    1. The **Underfit** model (red line) is too simple and misses the pattern in both training and testing data.
       - Training Error: {train_errors['Underfit (Linear)']:.4f}
       - Testing Error: {test_errors['Underfit (Linear)']:.4f}
    
    2. The **Good Fit** model (purple line) captures the general pattern without following noise.
       - Training Error: {train_errors['Good Fit']:.4f}
       - Testing Error: {test_errors['Good Fit']:.4f}
    
    3. The **Overfit** model (orange line) perfectly follows the training data, including noise.
       - Training Error: {train_errors['Overfit']:.4f}
       - Testing Error: {test_errors['Overfit']:.4f}
    
    Notice how the overfit model has the lowest training error but highest testing error - this is the hallmark of overfitting.
    """)

elif term_info['visual_type'] == 'underfitting':
    st.subheader("Visual Explanation: Underfitting")
    
    # Generate a synthetic dataset to demonstrate underfitting
    np.random.seed(0)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    y = 0.5 * x**2 + np.random.normal(0, 5, n_samples)  # Quadratic relationship with noise
    
    # Create DataFrame
    df_underfit = pd.DataFrame({'x': x, 'y': y})
    
    # Function to fit polynomial models of different degrees
    def fit_polynomial(x, y, degree):
        model = np.poly1d(np.polyfit(x, y, degree))
        x_line = np.linspace(min(x), max(x), 100)
        y_line = model(x_line)
        return x_line, y_line
    
    # Fit models of different complexity
    x_line_simple, y_line_simple = fit_polynomial(x, y, 1)    # Underfitting
    x_line_good, y_line_good = fit_polynomial(x, y, 2)        # Good fit
    x_line_complex, y_line_complex = fit_polynomial(x, y, 10)  # Overfitting
    
    # Visualization
    st.markdown("""
    **Underfitting** occurs when a model is too simple to capture the underlying pattern in the data.
    
    When a model underfits:
    - It performs poorly on both training and testing data
    - It fails to capture important relationships in the data
    - It has high bias (preconceived notion about the data)
    """)
    
    # Create plots
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Data Points',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Add model lines
    fig.add_trace(go.Scatter(
        x=x_line_simple, y=y_line_simple,
        mode='lines',
        name='Underfit (Linear)',
        line=dict(color='red', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_good, y=y_line_good,
        mode='lines',
        name='Good Fit (Quadratic)',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line_complex, y=y_line_complex,
        mode='lines',
        name='Overfit (Degree 10)',
        line=dict(color='orange', width=3)
    ))
    
    fig.update_layout(
        title='Demonstration of Underfitting vs. Good Fit vs. Overfitting',
        xaxis_title='x',
        yaxis_title='y',
        legend_title='Models',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Balancing Complexity: The Bias-Variance Tradeoff
    
    Machine learning involves finding the right balance between:
    
    - **Bias (Underfitting)**: When a model makes strong assumptions and is too simple
    - **Variance (Overfitting)**: When a model is too complex and captures noise
    
    The ideal model should:
    - Be complex enough to capture the underlying patterns
    - Be simple enough to generalize well to new data
    - Ignore the noise in the training data
    
    Techniques to prevent underfitting:
    - Use more complex models
    - Add more relevant features
    - Reduce regularization
    - Try different algorithms
    """)
    
    # Add a visual to explain bias-variance tradeoff
    bias_variance_img = "https://miro.medium.com/max/1400/1*WQXwEzJJHK7Q6RQFnQMJGw.png"
    st.image(bias_variance_img, caption="The Bias-Variance Tradeoff", width=700)

# Custom user example section
st.header("Try Your Own Example")

st.markdown("""
Now that you understand the key terminology, let's apply it to a house price prediction example using the California Housing dataset.
Adjust the features below and see how they affect the predicted house value.
""")

# Get California Housing dataset
california_df = datasets['California Housing']
X = california_df.drop('target', axis=1)
y = california_df['target']

# Train a model for demonstration
model = LinearRegression().fit(X, y)

# Create input form with California Housing features
col1, col2 = st.columns(2)

with col1:
    median_income = st.slider("Median Income (tens of thousands)", 
                           float(X['MedInc'].min()), 
                           float(X['MedInc'].max()), 
                           3.5)
    
    house_age = st.slider("Housing Median Age", 
                       float(X['HouseAge'].min()), 
                       float(X['HouseAge'].max()), 
                       28.0)
    
    avg_rooms = st.slider("Average Rooms per Household", 
                       float(X['AveRooms'].min()), 
                       float(X['AveRooms'].max()), 
                       5.0)
    
    avg_bedrooms = st.slider("Average Bedrooms per Household", 
                          float(X['AveBedrms'].min()), 
                          float(X['AveBedrms'].max()), 
                          1.0)

with col2:
    population = st.slider("Block Population", 
                        float(X['Population'].min()), 
                        float(X['Population'].max()), 
                        1500.0)
    
    avg_occupancy = st.slider("Average Household Occupancy", 
                           float(X['AveOccup'].min()), 
                           float(X['AveOccup'].max()), 
                           3.0)
    
    latitude = st.slider("Latitude", 
                      float(X['Latitude'].min()), 
                      float(X['Latitude'].max()), 
                      35.0)
    
    longitude = st.slider("Longitude", 
                       float(X['Longitude'].min()), 
                       float(X['Longitude'].max()), 
                       -120.0)

# Create new observation
new_district = pd.DataFrame({
    'MedInc': [median_income],
    'HouseAge': [house_age],
    'AveRooms': [avg_rooms],
    'AveBedrms': [avg_bedrooms],
    'Population': [population],
    'AveOccup': [avg_occupancy],
    'Latitude': [latitude],
    'Longitude': [longitude]
})

# Make prediction
predicted_value = model.predict(new_district)[0]

# Display results with ML terminology
st.subheader("ML Terminology Applied to This Example")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    **Dataset**: The California Housing dataset with information on housing districts
    
    **Features (X)**:
    - Median Income: ${median_income * 10000:.2f}
    - House Age: {house_age:.1f} years
    - Average Rooms: {avg_rooms:.2f} per household
    - Average Bedrooms: {avg_bedrooms:.2f} per household
    - Population: {population:.0f}
    - Average Occupancy: {avg_occupancy:.2f} persons per household
    - Location: ({latitude:.2f}, {longitude:.2f})
    
    **Observation**: This single housing district we're analyzing
    
    **Model**: Linear Regression that learned patterns from housing data
    """)

with col2:
    st.markdown(f"""
    **Target (y)**: Median House Value
    
    **Prediction**: ${predicted_value * 100000:,.2f}
    
    **Feature Importance**:
    - Median Income: ${model.coef_[0] * 100000:.2f} per unit
    - House Age: ${model.coef_[1] * 100000:.2f} per year
    - Average Rooms: ${model.coef_[2] * 100000:.2f} per room
    - Average Bedrooms: ${model.coef_[3] * 100000:.2f} per bedroom
    - Population: ${model.coef_[4] * 100000:.2f} per person
    - Average Occupancy: ${model.coef_[5] * 100000:.2f} per person
    - Latitude: ${model.coef_[6] * 100000:.2f} per degree
    - Longitude: ${model.coef_[7] * 100000:.2f} per degree
    """)

# Add a visual representation of the prediction
fig = go.Figure()

# Add base price (intercept)
fig.add_trace(go.Bar(
    x=['Base Value'],
    y=[model.intercept_ * 100000],  # Convert to dollars
    name='Base Value',
    marker_color='#4CAF50'
))

# Add feature contributions
features = X.columns
contributions = [model.coef_[i] * new_district[feature].values[0] * 100000 for i, feature in enumerate(features)]
colors = ['#2196F3', '#FF9800', '#9C27B0', '#E91E63', '#3F51B5', '#009688', '#FFC107', '#F44336']

for i, feature in enumerate(features):
    fig.add_trace(go.Bar(
        x=[feature],
        y=[contributions[i]],
        name=feature,
        marker_color=colors[i % len(colors)]
    ))

# Add total prediction
fig.add_trace(go.Bar(
    x=['Predicted Value'],
    y=[predicted_value * 100000],  # Convert to dollars
    name='Total',
    marker_color='#795548'
))

fig.update_layout(
    title='House Value Prediction Breakdown',
    xaxis_title='Components',
    yaxis_title='Value ($)',
    barmode='group',
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Additional explanation
st.markdown("""
This example illustrates how a machine learning model uses features (inputs) to make predictions about a target variable (output).
The model has learned patterns from many examples in the California Housing dataset to understand how features like median income
and house age affect median house values in different districts.
""")

# Add a glossary at the end
st.header("Machine Learning Terminology Glossary")

glossary_data = []
for term, info in ml_terminology.items():
    glossary_data.append({
        'Term': term,
        'Definition': info['definition']
    })

glossary_df = pd.DataFrame(glossary_data)
st.dataframe(glossary_df, use_container_width=True)
