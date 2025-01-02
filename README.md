# TrueTell - Real-time Misinformation Detection System

## 🌟 Overview
TrueTell is an advanced AI-powered real-time misinformation detection system developed for the "Hack the Hoax" hackathon. The system leverages cutting-edge Natural Language Processing (NLP), machine learning, and interactive visualization tools to detect, analyze, and alert users about potential misinformation in real-time.

## ✨ Key Features

### 1. Real-time Content Analysis
- Live content stream processing and analysis
- Multi-model classification system
- Sentiment analysis integration
- Entity recognition and relationship mapping

### 2. Advanced Detection Capabilities
- Pattern recognition in narratives
- Source credibility verification
- Temporal analysis of information spread
- Entity manipulation detection

### 3. Interactive Dashboard
- Real-time monitoring interface
- Dynamic visualization of analysis results
- Risk score tracking
- Source credibility metrics

### 4. Knowledge Graph Integration
- Entity relationship mapping
- Pattern detection visualization
- Temporal analysis of narrative evolution
- Interactive graph exploration

### 5. Alert System
- Real-time notification system
- Configurable alert thresholds
- Multi-channel notifications (Webhook, Slack)
- Alert management interface

### 6. Integration Layer
- External API integration
- Webhook support
- Slack notifications
- Extensible integration framework

## 🛠️ Technology Stack

### Core Technologies
- **Programming Language**: Python 3.9+
- **Web Framework**: Streamlit
- **ML/NLP Libraries**: 
  - Hugging Face Transformers
  - SpaCy
  - NLTK
  - TextBlob
  - Scikit-learn

### Data Processing
- Pandas
- NumPy
- NetworkX (for knowledge graphs)

### Visualization
- Plotly
- Matplotlib
- Seaborn

### External APIs
- Google Fact Check API
- News API
- Twitter API
- Reddit API

## 📊 Dataset

### Data Collection
1. Collected 1,589+ tweets related to misinformation
2. Converted tweets to images
3. Used OCR (Optical Character Recognition) to extract text
4. Created structured dataset for training

### Data Processing Pipeline
1. **Image Text Extraction** (`extract_from_image.py`):
   - Uses Tesseract OCR
   - Processes multiple image formats
   - Outputs structured Excel file

2. **Dataset Creation** (`dataset_make.py`):
   - Text cleaning and preprocessing
   - Label encoding
   - Duplicate removal
   - Dataset validation

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/TrueTell.git
cd TrueTell
```

### Step 2: Create and Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Setup
Create a `.env` file in the root directory with the following configurations:
```plaintext
# API Keys
GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key
NEWS_API_KEY=your_news_api_key
GUARDIAN_API_KEY=your_guardian_api_key

# Webhook Configuration
WEBHOOK_URL=your_webhook_url

# Slack Integration
SLACK_TOKEN=your_slack_token
SLACK_CHANNEL=your_slack_channel

# Twitter Credentials
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret

# Reddit Credentials
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

### Step 5: Initialize the Model
```bash
python src/train.py
```

### Step 6: Run the Application
```bash
streamlit run src/app.py
```

## 📁 Project Structure
```
TrueTell/
├── src/
│   ├── broadcast/                 # Broadcast processing modules
│   │   ├── __init__.py
│   │   ├── analyzer.py           # Broadcast content analysis
│   │   ├── social_monitor.py     # Social media monitoring
│   │   ├── sources.py           # News source handling
│   │   └── stream.py            # Stream processing
│   ├── alert_system.py          # Alert management
│   ├── app.py                   # Main Streamlit application
│   ├── credibility_scorer.py    # Content credibility scoring
│   ├── dashboard.py             # Dashboard visualization
│   ├── fact_checker.py          # Fact checking logic
│   ├── integration_layer.py     # External integrations
│   ├── knowledge_graph.py       # Knowledge graph management
│   ├── model_trainer.py         # ML model training
│   ├── source_checker.py        # Source verification
│   ├── train.py                # Model training script
│   └── utils.py                # Utility functions
├── datasets/                    # Dataset storage
├── models/                      # Trained model storage
│   └── visualizations/         # Model performance visualizations
├── logs/                       # Application logs
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🎯 Features in Detail

### 1. Real-time Content Analysis
- **Multi-model Classification**: Uses ensemble of models for robust classification
- **Sentiment Analysis**: Analyzes emotional tone and bias
- **Entity Recognition**: Identifies and tracks key entities
- **Pattern Detection**: Recognizes suspicious narrative patterns

### 2. Knowledge Graph
- **Entity Tracking**: Maps relationships between entities
- **Temporal Analysis**: Tracks information evolution
- **Pattern Recognition**: Identifies misinformation patterns
- **Interactive Visualization**: Explore entity relationships

### 3. Alert System
- **Configurable Thresholds**: Customize alert sensitivity
- **Multi-channel Notifications**: Slack, Webhook support
- **Alert Management**: Track and manage alerts
- **Risk Assessment**: Detailed risk scoring

### 4. Dashboard
- **Real-time Monitoring**: Live content analysis
- **Interactive Visualizations**: Dynamic data exploration
- **Performance Metrics**: Track system performance
- **Alert Management**: Handle and respond to alerts

## 🔄 Model Training Process

### 1. Data Preparation
- Text extraction from images
- Data cleaning and preprocessing
- Label encoding
- Dataset validation

### 2. Model Architecture
- TF-IDF vectorization
- RandomForest classifier
- Grid search optimization
- Cross-validation

### 3. Performance Metrics
- Accuracy scoring
- Confusion matrix
- ROC curves
- Feature importance

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.