# TruthTell - Real-time Misinformation Detection System

## 🛠️ Overview
TruthTell is an AI-powered real-time misinformation detection system developed during the **TruthTell Hackathon** for the **Create in India Challenge**. The system is designed to detect, flag, and alert users about misinformation in live broadcasts, ensuring a more informed and aware audience.

## ✨ Features
- **Real-time Content Analysis**: Analyze and process live content in real-time.
- **AI-Powered Detection**: Advanced NLP techniques to detect misinformation.
- **Interactive Dashboard**: Visualize insights with dynamic charts and graphs.
- **Multi-Channel Alert System**: Alerts via email, SMS, or push notifications.
- **Fact-Checking Integration**: Leverage external fact-checking APIs for accuracy.
- **Source Credibility Verification**: Score and verify the credibility of sources.

## 🖥️ Tech Stack
- **Programming Language**: Python 3.9+
- **Frontend**: Streamlit
- **NLP Frameworks**: NLTK, Transformers
- **Visualization**: Plotly
- **Database**: SQLite
- **Other Libraries**: Various AI/ML frameworks

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
      ```
    Replace `your_google_fact_check_api_key` and `your_news_api_key` with your actual API keys.

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
│   ├── realtime_processor.py  # Real-time content processing
│   ├── source_checker.py      # Source credibility verification
│   └── utils.py               # Utility functions
├── .env                       # Environment variables (create this file)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🤝 Contributing
We welcome contributions to **TruthTell**! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.


## 📜 License
This project is licensed under the **[Your License Name]**. See the [LICENSE](LICENSE) file for more details.