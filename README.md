# TruthTell - Real-time Misinformation Detection System

## 🛠️ Overview
TruthTell is an AI-powered real-time misinformation detection system developed during the **TruthTell Hackathon** for the **Create in India Challenge**. The system is designed to detect, flag, and alert users about misinformation in live broadcasts, ensuring a more informed and aware audience. TruthTell leverages advanced Natural Language Processing (NLP), machine learning, and interactive visualization tools to provide actionable insights in real time.

## ✨ Features
- **Real-time Content Analysis**: Analyze and process live content streams.
- **AI-Powered Detection**: Advanced NLP and multi-model classification for misinformation detection.
- **Interactive Dashboard**: Monitor trends, risk scores, and source credibility using dynamic visualizations.
- **Knowledge Graph Visualization**: Map relationships, detect narrative patterns, and explore temporal content connections.
- **Multi-Channel Alert System**: Get notifications via email, Slack, or webhooks.
- **Fact-Checking Integration**: Utilize APIs like Google Fact Check and News API for accuracy.
- **Source Credibility Verification**: Score and track the reliability of information sources.

## 🖥️ Tech Stack
- **Programming Language**: Python 3.9+
- **Frontend**: Streamlit
- **Graph Processing**: NetworkX
- **Visualization**: Plotly
- **NLP Frameworks**: Hugging Face Transformers, SpaCy, NLTK, TextBlob
- **Data Processing**: Pandas, NumPy
- **APIs**: Twitter API, News API, Google Fact Check API
- **Other Libraries**: AsyncIO, Custom REST APIs

## 🚀 Installation
Follow these steps to set up the project locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/XENO2410/truthtell.git
    cd truthtell
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your `.env` file**:
    - Navigate to the main folder (`TRUETELL`).
    - Create a `.env` file in the root directory of the project (beside the `src` folder).
    - Add the following keys to your `.env` file:
      ```env
      GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key
      NEWS_API_KEY=your_news_api_key
      TWITTER_BEARER_TOKEN=your_twitter_bearer_token
      ```
    Replace `your_google_fact_check_api_key`, `your_news_api_key`, and `your_twitter_bearer_token` with your actual API keys.

5. **Run the application**:
    ```bash
    streamlit run src/app.py
    ```

## 📁 Project Structure
```plaintext
TRUETELL/
├── src/
│   ├── alert_system.py        # Alert management
│   ├── app.py                 # Main application
│   ├── credibility_scorer.py  # Content scoring
│   ├── dashboard.py           # Visualizations
│   ├── fact_checker.py        # Fact-checking logic
│   ├── fact_database.py       # Data storage and retrieval
│   ├── integration_layer.py   # External service integrations
│   ├── knowledge_graph.py     # Graph visualizations
│   ├── realtime_processor.py  # Real-time content processing
│   ├── source_checker.py      # Source credibility verification
│   ├── social_monitor.py      # Social media monitoring
│   └── utils.py               # Utility functions
├── .env                       # Environment variables (create this file)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🤝 Contributing
We welcome contributions to **TruthTell**! Follow these steps to contribute:

1. **Fork the repository**:
    ```bash
    git clone https://github.com/your-username/truthtell.git
    ```
2. **Create a feature branch**:
    ```bash
    git checkout -b feature-name
    ```
3. **Commit your changes**:
    ```bash
    git commit -m "Add some feature"
    ```
4. **Push to the branch**:
    ```bash
    git push origin feature-name
    ```
5. **Open a Pull Request**: Go to the original repository and open a pull request.

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
