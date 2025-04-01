# Multiclass Classification for Product Categorization - Streamlit App (Fixed)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(
    page_title="Product Categorization", 
    page_icon="🛒",
    layout="wide"
)

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')

download_nltk_resources()

# Class to extract text features
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.field].values

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Function to generate synthetic product data
@st.cache_data
def generate_product_data(n_samples=2000):
    np.random.seed(42)
    
    # Define product categories and their properties
    categories = {
        'Electronics': {
            'keywords': ['device', 'electronic', 'tech', 'gadget', 'digital', 'wireless', 'smart', 'battery', 
                        'charger', 'screen', 'computer', 'laptop', 'phone', 'tablet', 'camera', 'headphone', 
                        'speaker', 'wifi', 'bluetooth', 'gaming'],
            'price_range': (50, 2000),
            'weight_range': (0.1, 10),
            'rating_mean': 4.1
        },
        'Clothing': {
            'keywords': ['apparel', 'wear', 'fashion', 'cloth', 'garment', 'outfit', 'dress', 'shirt', 't-shirt', 
                         'pants', 'jeans', 'jacket', 'coat', 'sweater', 'hoodie', 'shoes', 'socks', 'hat', 'style', 
                         'cotton', 'wool', 'leather', 'casual', 'formal'],
            'price_range': (10, 200),
            'weight_range': (0.1, 2),
            'rating_mean': 3.9
        },
        'Home & Kitchen': {
            'keywords': ['home', 'house', 'kitchen', 'cookware', 'appliance', 'furniture', 'decor', 'bedding', 
                         'utensil', 'pan', 'pot', 'knife', 'plate', 'bowl', 'cup', 'glass', 'chair', 'table', 
                         'bed', 'sofa', 'lamp', 'curtain', 'rug', 'pillow'],
            'price_range': (15, 500),
            'weight_range': (0.2, 25),
            'rating_mean': 4.0
        },
        'Books': {
            'keywords': ['book', 'read', 'novel', 'fiction', 'nonfiction', 'textbook', 'story', 'author', 'page', 
                         'chapter', 'cover', 'paperback', 'hardcover', 'bestseller', 'literature', 'biography', 
                         'mystery', 'fantasy', 'comic', 'educational'],
            'price_range': (5, 50),
            'weight_range': (0.2, 3),
            'rating_mean': 4.3
        },
        'Sports & Outdoors': {
            'keywords': ['sport', 'outdoor', 'fitness', 'exercise', 'athletic', 'game', 'recreation', 'camping', 
                         'hiking', 'biking', 'swimming', 'running', 'ball', 'racket', 'gym', 'workout', 'gear', 
                         'equipment', 'training', 'adventure'],
            'price_range': (10, 400),
            'weight_range': (0.1, 15),
            'rating_mean': 4.0
        }
    }
    
    # Common product adjectives and descriptors
    adjectives = ['premium', 'quality', 'best', 'new', 'professional', 'portable', 'lightweight', 'durable', 
                 'comfortable', 'advanced', 'modern', 'stylish', 'classic', 'unique', 'essential', 'perfect', 
                 'innovative', 'compact', 'high-performance', 'reliable', 'affordable', 'luxury']
    
    # Generate data
    data = []
    for _ in range(n_samples):
        # Randomly choose a category
        category = np.random.choice(list(categories.keys()))
        cat_props = categories[category]
        
        # Generate price, weight, and rating based on category properties
        price = np.random.uniform(*cat_props['price_range'])
        weight = np.random.uniform(*cat_props['weight_range'])
        rating = min(5, max(1, np.random.normal(cat_props['rating_mean'], 0.5)))
        
        # Generate a product title
        keywords = cat_props['keywords']
        num_keywords = np.random.randint(1, 4)
        title_keywords = np.random.choice(keywords, num_keywords, replace=False)
        title_adj = np.random.choice(adjectives, np.random.randint(0, 3), replace=False)
        brand = f"Brand{np.random.randint(1, 31)}"
        
        # Construct title with varying patterns
        pattern = np.random.randint(1, 4)
        if pattern == 1:
            title = f"{brand} {' '.join(title_adj)} {' '.join(title_keywords)}"
        elif pattern == 2:
            title = f"{' '.join(title_adj)} {' '.join(title_keywords)} by {brand}"
        else:
            title = f"{' '.join(title_keywords)} {' '.join(title_adj)} - {brand}"
        
        # Generate a product description
        desc_length = np.random.randint(10, 30)
        desc_keywords = np.random.choice(keywords, min(desc_length, len(keywords)), replace=True)
        desc_adj = np.random.choice(adjectives, min(desc_length // 3, len(adjectives)), replace=True)
        
        description_words = list(desc_keywords) + list(desc_adj)
        np.random.shuffle(description_words)
        description = ' '.join(description_words[:desc_length])
        description = description.capitalize() + '.'
        
        # Add some random connectors and phrases to make the description more realistic
        phrases = [
            "Perfect for everyday use.",
            "Great value for money.",
            "Highly recommended by customers.",
            "Best seller in its category.",
            "Limited edition item.",
            "Available in multiple colors.",
            "Exclusive to our store.",
            "Made with premium materials.",
            "Designed for maximum comfort."
        ]
        
        description += ' ' + np.random.choice(phrases)
        
        # Generate stock level
        stock = np.random.randint(0, 100)
        
        # Add discount information (some products have discount, some don't)
        has_discount = np.random.choice([True, False], p=[0.3, 0.7])
        discount_percent = np.random.randint(5, 50) if has_discount else 0
        
        # Add seller information
        seller_rating = min(5, max(1, np.random.normal(4, 0.7)))
        seller_response_time = np.random.randint(1, 72)  # hours
        
        # Add review count
        review_count = np.random.randint(0, 1000)
        
        data.append({
            'title': title,
            'description': description,
            'price': price,
            'weight': weight,
            'rating': rating,
            'category': category,
            'stock': stock,
            'discount_percent': discount_percent,
            'seller_rating': seller_rating,
            'seller_response_time': seller_response_time,
            'review_count': review_count
        })
    
    return pd.DataFrame(data)

# Function to train models
@st.cache_resource
def train_models(data):
    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['category'])
    
    # Define numeric features
    numeric_features = ['price', 'weight', 'rating', 'stock', 'discount_percent', 
                        'seller_rating', 'seller_response_time', 'review_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
    
    # Create pipelines that work directly with the original DataFrame
    # Pipeline for Random Forest, SVM, and Logistic Regression
    pipeline_others = Pipeline([
        ('features', FeatureUnion([
            ('numeric', Pipeline([
                ('selector', ColumnTransformer([
                    ('num', MinMaxScaler(), numeric_features)
                ], remainder='drop')),
            ])),
            ('text_title', Pipeline([
                ('selector', TextSelector('title')),
                ('tfidf', TfidfVectorizer(max_features=1000))
            ])),
            ('text_desc', Pipeline([
                ('selector', TextSelector('description')),
                ('tfidf', TfidfVectorizer(max_features=1000))
            ]))
        ]))
    ])
    
    # Pipeline specifically for Naive Bayes (using CountVectorizer)
    pipeline_nb = Pipeline([
        ('features', FeatureUnion([
            ('numeric', Pipeline([
                ('selector', ColumnTransformer([
                    ('num', MinMaxScaler(), numeric_features)
                ], remainder='drop')),
            ])),
            ('text_title', Pipeline([
                ('selector', TextSelector('title')),
                ('count', CountVectorizer(max_features=1000))
            ])),
            ('text_desc', Pipeline([
                ('selector', TextSelector('description')),
                ('count', CountVectorizer(max_features=1000))
            ]))
        ]))
    ])
    
    # Define models
    models = {
        'Random Forest': Pipeline([
            ('preproc', pipeline_others),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('preproc', pipeline_others),
            ('classifier', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ]),
        'Logistic Regression': Pipeline([
            ('preproc', pipeline_others),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Naive Bayes': Pipeline([
            ('preproc', pipeline_nb),
            ('classifier', MultinomialNB())
        ])
    }
    
    # Train each model and store results
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    # Sort models by accuracy
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['accuracy'], reverse=True)}
    
    return results, label_encoder, X_test, y_test

# Generate or load data
product_data = generate_product_data()

# Train models
with st.spinner('Training models... This might take a minute.'):
    model_results, label_encoder, X_test, y_test = train_models(product_data)

# Main application
st.title("🛒 Product Categorization with Multiclass Classification")
st.write("""
This application demonstrates multiclass classification in machine learning by categorizing
products into different departments based on their features and descriptions.
""")

# Create tabs for the application
tab1, tab2, tab3 = st.tabs(["Predict Product Category", "Model Performance", "Data Exploration"])

with tab1:
    st.header("Product Category Predictor")
    
    st.write("""
    Enter product details to predict which category it belongs to.
    """)
    
    col1, col2 = st.columns([3, 2])
    input_data = pd.DataFrame({})
    

    
    with col2:
        # Numeric inputs
        price = st.number_input("Price ($)", min_value=0.0, max_value=5000.0, value=99.99)
        weight = st.number_input("Weight (kg)", min_value=0.01, max_value=100.0, value=0.3)
        rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        stock = st.number_input("Stock Level", min_value=0, max_value=1000, value=45)
        discount = st.slider("Discount (%)", min_value=0, max_value=90, value=10)
        seller_rating = st.slider("Seller Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
        seller_response = st.number_input("Seller Response Time (hours)", min_value=1, max_value=72, value=24)
        review_count = st.number_input("Number of Reviews", min_value=0, max_value=10000, value=120)
    
    with col1:
        # Text inputs
        product_title = st.text_input("Product Title", "Wireless Bluetooth Headphones with Noise Cancellation")
        product_desc = st.text_area("Product Description", 
            "Premium quality headphones with active noise cancellation technology. " + 
            "Long battery life and comfortable ear cups for extended use. " + 
            "Compatible with all Bluetooth devices.")

        # Choose model
        selected_model = st.selectbox("Select Model for Prediction", list(model_results.keys()))
        
        submitted = st.button("Predict Category", type="primary")
    
    if submitted:
        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'title': [product_title],
            'description': [product_desc],
            'price': [price],
            'weight': [weight],
            'rating': [rating],
            'stock': [stock],
            'discount_percent': [discount],
            'seller_rating': [seller_rating],
            'seller_response_time': [seller_response],
            'review_count': [review_count]
        })
    
        # Get model
        model = model_results[selected_model]['model']
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get predicted category
        category = label_encoder.inverse_transform([prediction])[0]
        
        # Display result
        st.subheader("Prediction Result")
        
        # Display category with emoji
        category_emojis = {
            'Electronics': '🔌',
            'Clothing': '👕',
            'Home & Kitchen': '🍳',
            'Books': '📚',
            'Sports & Outdoors': '⚽'
        }
        
        emoji = category_emojis.get(category, '🛒')
        
        st.markdown(f"<h2 style='text-align: center;'>{emoji} {category}</h2>", unsafe_allow_html=True)
        
        # Display probabilities
        st.subheader("Category Probabilities")
        
        proba_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = sns.barplot(x='Probability', y='Category', data=proba_df, ax=ax, 
                          palette='viridis')
        
        # Add percentage labels to the bars
        for i, p in enumerate(bars.patches):
            width = p.get_width()
            ax.text(width + 0.02, p.get_y() + p.get_height()/2, 
                    f'{width:.1%}', ha='left', va='center')
        
        plt.xlim(0, 1)
        plt.title('Probability by Category')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance analysis if it's Random Forest
        if selected_model == 'Random Forest':
            st.subheader("Feature Importance")
            st.info("Feature importance is available for Random Forest, but detailed feature names are not available in this simplified model.")

with tab2:
    st.header("Model Performance")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Create a DataFrame for model comparison
    model_comparison = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [results['accuracy'] for results in model_results.values()]
    })
    
    # Display the comparison table
    st.dataframe(model_comparison.style.highlight_max(subset=['Accuracy']))
    
    # Plot the accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='Accuracy', data=model_comparison, ax=ax)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy Score')
    plt.ylim(0, 1)
    st.pyplot(fig)
    
    # Select model for detailed performance analysis
    model_for_analysis = st.selectbox("Select model for detailed analysis", list(model_results.keys()))
    
    # Get the data for the selected model
    selected_data = model_results[model_for_analysis]
    y_true = selected_data['y_test']
    y_pred = selected_data['y_pred']
    
    # Display classification report
    st.subheader("Classification Report")
    
    # Get the classification report as a dictionary
    report = classification_report(y_true, y_pred, output_dict=True, 
                                 target_names=label_encoder.classes_)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Display the report
    st.dataframe(report_df.style.format("{:.2f}"))
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a DataFrame for better labeling
    cm_df = pd.DataFrame(cm, 
                       index=label_encoder.classes_,
                       columns=label_encoder.classes_)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

with tab3:
    st.header("Dataset Exploration")
    
    # Display dataset info
    st.subheader("Dataset Overview")
    st.write(f"Number of products: {len(product_data)}")
    st.write(f"Number of categories: {product_data['category'].nunique()}")
    
    # Show sample of the dataset
    st.subheader("Sample Products")
    st.dataframe(product_data[['title', 'price', 'rating', 'category']].sample(5))
    
    # Distribution of categories
    st.subheader("Category Distribution")
    
    # Create a DataFrame for the category counts
    category_counts = product_data['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Plot the distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Category', y='Count', data=category_counts, ax=ax)
    plt.title('Number of Products by Category')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Price distribution by category
    st.subheader("Price Distribution by Category")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='category', y='price', data=product_data, ax=ax)
    plt.title('Price Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Rating distribution
    st.subheader("Rating Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(product_data['rating'], bins=10, kde=True, ax=ax)
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(fig)
    
    # Common words analysis
    st.subheader("Most Common Words by Category")
    
    category_to_explore = st.selectbox("Select category to explore", product_data['category'].unique())
    
    # Filter data for the selected category
    category_data = product_data[product_data['category'] == category_to_explore]
    
    # Extract and preprocess text
    title_text = ' '.join([preprocess_text(title) for title in category_data['title']])
    desc_text = ' '.join([preprocess_text(desc) for desc in category_data['description']])
    category_text = title_text + ' ' + desc_text
    
    try:
        from wordcloud import WordCloud
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(category_text)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'Most Common Words in {category_to_explore} Products')
        ax.axis("off")
        st.pyplot(fig)
    except ImportError:
        # As a fallback, show most common words in a bar chart
        from collections import Counter
        import re
        
        # Count word frequencies
        words = re.findall(r'\w+', category_text.lower())
        word_counts = Counter(words).most_common(20)
        
        # Create DataFrame
        word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Count', y='Word', data=word_df, ax=ax)
        plt.title(f'Most Common Words in {category_to_explore} Products')
        st.pyplot(fig)

# Add a sidebar with additional information
with st.sidebar:
    st.title("About")
    st.info("""
    **Product Categorization App**
    
    This application demonstrates multiclass classification in machine learning 
    by categorizing products into different departments based on their features 
    and descriptions.
    
    **Categories:**
    - Electronics
    - Clothing
    - Home & Kitchen
    - Books
    - Sports & Outdoors
    
    **Models Demonstrated:**
    - Random Forest
    - Support Vector Machines (SVM)
    - Logistic Regression
    - Naive Bayes
    """)
    
    st.subheader("How Multiclass Classification Works")
    st.write("""
    Multiclass classification extends binary classification to handle multiple categories:
    
    1. **One-vs-Rest (OVR)**: Trains a binary classifier for each class against all others
    2. **One-vs-One (OVO)**: Trains a classifier for each pair of classes
    
    The models use features extracted from:
    - Product titles and descriptions (using TF-IDF or Count vectorization)
    - Numeric attributes (price, weight, rating, etc.)
    
    The final prediction assigns products to the most likely category.
    """)
    
    # Allow downloading the dataset
    st.subheader("Download Dataset")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(product_data)
    st.download_button(
        "Download Product Dataset",
        csv,
        "product_data.csv",
        "text/csv",
        key='download-csv'
    )
