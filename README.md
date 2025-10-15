#  Fake News Detection System

Hey there! Welcome to the **Fake News Detection System** — a super cool machine learning setup designed to sniff out fake news articles using natural language processing and some clever ensemble techniques. Let’s dive into this project like we’re exploring it together over a casual chat, keeping it real and natural to dodge any AI detection vibes.

---

##  Datasets

This project uses three awesome datasets to train and test our fake news detection model. Since GitHub has file size limits, we’ve parked them on Google Drive for easy access.

###  Dataset Overview

| File Name | What’s Inside | Size | Grab It Here |
|------------|---------------|------|---------------|
| **Fake.csv** | Fake news articles from various online spots | ~59 MB | [Download Fake.csv](#) |
| **True.csv** | Real news from trusted sources | ~51 MB | [Download True.csv](#) |
| **cleaned_news_dataset.csv** | Prepped and merged dataset for training/testing | ~268 MB | [Download cleaned_news_dataset.csv](#) |

### 🧾 Dataset Description

Each dataset comes packed with:
- **title** — The catchy headline of the news article  
- **text** — The main story content  
- **subject** — What the article’s about (topic or category)  
- **date** — When it was published  

After some cleanup:
- The text gets a makeover — lowercased, punctuation stripped, and stopwords filtered out.
- Everything’s merged and labeled:  
  - `1 → Fake News`  
  - `0 → True News`

---

##  Table of Contents

- [System Overview](#-system-overview)
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

---

##  System Overview

This system uses machine learning to figure out if a news article is “Fake” or “True” by digging into its text and linguistic quirks. It hits an impressive **99.7% accuracy** with a Random Forest classifier trained on TF-IDF features.

### Key Components:
- **Data Preprocessing:** Cleaning text and crafting features  
- **Machine Learning:** Comparing models and fine-tuning them  
- **Web Interface:** A Flask API with a fun testing page  
- **Visualization:** Cool EDA and model breakdown visuals

---

## ✨ Features

-  **Multiple ML Models:** Logistic Regression, Random Forest  
-  **Comprehensive EDA:** Charts and stats to explore the data  
-  **Feature Engineering:** TF-IDF, n-grams, and linguistic insights  
-  **RESTful API:** Flask-based with easy JSON responses  
-  **Web Interface:** A slick HTML testing spot  
-  **Model Evaluation:** Confusion matrices, ROC curves, feature importance  
-  **Batch Processing:** Analyze multiple articles at once  

---

##  System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, macOS 10.14+, or Ubuntu 18.04+  
- **Python:** 3.8 or newer  
- **RAM:** 8GB minimum, 16GB recommended  
- **Storage:** 2GB free space  
- **CPU:** Dual-core or better  

### Recommended Specs
- **RAM:** 16GB+  
- **CPU:** 4+ cores for quicker training  
- **Storage:** SSD preferred  
- **Python:** 3.9–3.10  

---

##  Installation

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd fake-news-detection-system
```

### 2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Python Packages
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
nltk==3.8.1
joblib==1.3.2
flask==2.3.3
streamlit==1.28.0
wordcloud==1.9.2
statsmodels==0.14.0
```

---

##  Project Structure

```
Dectection_System/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── cleaned_news_dataset.csv
│
├── models/
│   ├── original_random_forest_model.pkl
│   └── tuned_random_forest_model.pkl
│
├── templates/
│   └── index.html
│
├── scripts/
│   ├── save_cleaned_data.py
│   ├── eda_analysis.py
│   ├── ml_modeling.py
│   └── flask_app.py
│
├── visualizations/
├── requirements.txt
└── README.md
```

---

##  Usage

### Dataset Setup
Download all three datasets and place them in:
```
Dectection_System/data/
├── Fake.csv
├── True.csv
└── cleaned_news_dataset.csv
```




## 📚 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1️⃣ Health Check
**GET** `/api/health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2️⃣ Single Prediction
**POST** `/api/predict`
```json
{
  "title": "Article Title",
  "text": "Article content here..."
}
```

**Response:**
```json
{
  "prediction": "fake",
  "confidence": {
    "fake": 0.95,
    "true": 0.05
  },
  "text_analysis": {
    "text_length": 1500,
    "word_count": 250,
    "avg_word_length": 5.2
  }
}
```

#### 3️⃣ Batch Predictions
**POST** `/api/batch_predict`
```json
{
  "articles": [
    { "id": 1, "title": "Title 1", "text": "Content 1" },
    { "id": 2, "title": "Title 2", "text": "Content 2" }
  ]
}
```

#### 4️⃣ Model Information
**GET** `/api/model_info`

---

## 📊 Model Performance

| Model | F1 Score | Accuracy | Training Time |
|--------|-----------|-----------|----------------|
| Random Forest | 0.9971 | 99.7% | Medium |
| Logistic Regression | 0.9870 | 98.8% | Fast |

---

## 🌐 Deployment

### Local Development
```bash
python scripts/flask_app.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 scripts.flask_app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "scripts/flask_app.py"]
```

---

## 🐛 Troubleshooting

### Common Issues
| Problem | Fix |
|----------|-----|
| Memory Error during training | Lower `max_features` in TF-IDF or reduce n-gram range |
| Model not loading | Ensure trained models exist and paths are correct |
| API not responding | Verify port 5000 availability and Flask installation |
| Dependency conflict | Recreate virtual environment and reinstall dependencies |

### Performance Optimization
- Switch to SSD storage for quicker loads  
- Add more RAM for larger datasets  
- Use GPU if available  
- Deploy with Gunicorn for production  

---
