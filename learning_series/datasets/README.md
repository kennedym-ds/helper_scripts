# Learning Series Datasets

This directory contains sample datasets used throughout the machine learning learning series. These datasets are carefully curated to provide realistic examples for different types of machine learning problems.

## Dataset Categories

### 1. Beginner-Friendly Datasets
- **iris.csv**: Classic iris flower classification dataset
- **boston_housing.csv**: House price prediction dataset
- **customer_data.csv**: Customer segmentation and behavior analysis

### 2. Computer Vision Datasets
- **mnist_sample.csv**: Handwritten digit recognition (subset)
- **cifar10_sample/**: Small sample of CIFAR-10 images
- **faces_sample/**: Sample face images for detection exercises

### 3. Natural Language Processing
- **movie_reviews.csv**: Movie review sentiment analysis
- **news_articles.csv**: Text classification and topic modeling
- **chatbot_conversations.csv**: Dialogue and conversation data

### 4. Time Series Data
- **stock_prices.csv**: Historical stock price data
- **weather_data.csv**: Daily weather measurements
- **sales_forecast.csv**: Monthly sales figures for forecasting

### 5. Advanced Datasets
- **medical_images/**: Sample medical scans (anonymized)
- **audio_samples/**: Speech and sound classification data
- **synthetic_gan.csv**: Generated data for GAN training examples

## Dataset Descriptions

### customer_data.csv
**Purpose**: Customer analysis and segmentation  
**Features**: age, income, spending_score, education_level, city, purchase_frequency  
**Size**: 1000 rows  
**Use Cases**: Clustering, classification, regression  

### iris.csv
**Purpose**: Multi-class classification  
**Features**: sepal_length, sepal_width, petal_length, petal_width, species  
**Size**: 150 rows  
**Use Cases**: Classification, feature selection, visualization  

### movie_reviews.csv
**Purpose**: Sentiment analysis  
**Features**: review_text, sentiment, rating, genre  
**Size**: 5000 rows  
**Use Cases**: Text preprocessing, classification, NLP  

## Usage Guidelines

1. **Start Simple**: Begin with iris.csv for basic classification concepts
2. **Progress Gradually**: Move to more complex datasets as skills develop
3. **Real-World Practice**: Use customer_data.csv for realistic business scenarios
4. **Domain-Specific**: Choose datasets relevant to your interests

## Data Generation Scripts

The `generate_datasets.py` script can create additional synthetic datasets:

```python
python utils/generate_datasets.py --dataset customer --size 1000
python utils/generate_datasets.py --dataset time_series --days 365
```

## Ethical Considerations

- All datasets are either public domain or synthetically generated
- No personal or sensitive information is included
- Datasets are designed for educational purposes only
- Some datasets may contain historical biases - use them as learning opportunities

## Contributing

To add new datasets:
1. Ensure data is appropriate for educational use
2. Include comprehensive documentation
3. Provide example usage code
4. Test with multiple learning modules

## Data Sources

- **Public Datasets**: UCI ML Repository, Kaggle public datasets
- **Synthetic Data**: Generated using statistical models and domain knowledge
- **Educational Sets**: Curated specifically for learning objectives

Remember: The goal is learning, not achieving perfect accuracy. Focus on understanding the process and concepts rather than optimizing for the highest scores.