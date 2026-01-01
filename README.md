# AI for IT Support: Automated Ticket Classification

**Copyright (c) 2026 Shrikara Kaudambady**

This project demonstrates a practical application of AI and Machine Learning for IT support operations. It provides a Jupyter notebook that builds and trains a Natural Language Processing (NLP) model to automatically classify IT support tickets into predefined categories (`Hardware`, `Software`, `Network`) based on their text descriptions.

## Solution Explanation

Manually sorting and routing IT support tickets is a significant operational bottleneck for many organizations. It's repetitive, time-consuming, and can lead to delays in ticket resolution. This project presents an automated solution using classic NLP techniques.

The workflow implemented in the notebook is as follows:

### 1. Dataset
The project uses a synthetic dataset named `it_support_tickets.csv`, which contains two columns:
- `description`: The text of the support request from a user.
- `category`: The correct IT category for that ticket.

### 2. NLP Pipeline
To prepare the text data for the machine learning model, we perform several preprocessing steps:
- **Text Cleaning:** The raw text is standardized by converting it to lowercase and removing all punctuation and numbers.
- **Stopword Removal:** Common English words that provide little semantic meaning (e.g., "is", "the", "a", "in") are removed.
- **Stemming:** Words are reduced to their root form (e.g., "connecting", "connected" both become "connect"). This helps the model treat different forms of the same word as equivalent.

### 3. Feature Engineering with TF-IDF
Machine learning models require numerical input. We use the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization technique to convert the cleaned text into a matrix of numerical features. TF-IDF is powerful because it emphasizes words that are important to a specific document, not just words that are common across all documents.

### 4. Model Training and Evaluation
- **Model:** A `LogisticRegression` classifier is trained on the TF-IDF features. This model is chosen because it is a highly effective and interpretable baseline for text classification tasks.
- **Evaluation:** The model's performance is measured on an unseen test set. The notebook calculates the overall accuracy and generates a detailed `classification_report` (showing precision, recall, and F1-score for each category) and a `confusion_matrix` to visualize where the model makes mistakes.

## How to Use

Follow these steps to set up your environment and run the ticket classification notebook.

### 1. Clone the Repository
If this project is on a Git repository, clone it to your local machine:
```bash
git clone <repository-url>
cd gemini-cli-projects/it-support-ticket-classification
```

### 2. Create a Virtual Environment
Using a virtual environment is best practice for managing project-specific dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
The notebook uses the NLTK library to remove stopwords. You need to download this data once. Run the following command in your terminal:
```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

### 5. Launch Jupyter
Start the Jupyter Notebook server from your terminal.
```bash
jupyter notebook
```
This will open a new tab in your web browser showing the project directory.

### 6. Run the Notebook
Click on the `it_ticket_classification.ipynb` file to open it. You can then execute the cells one by one to see the entire process, from data loading to predicting the category of new tickets.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
