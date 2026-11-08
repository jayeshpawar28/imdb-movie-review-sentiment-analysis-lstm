# IMDB Movie Review Sentiment Analysis using LSTM

A beginner-friendly Natural Language Processing (NLP) project that classifies IMDB movie reviews as **positive** or **negative** using an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras.

## Project Overview

This project uses the IMDB movie reviews dataset to demonstrate a basic text-classification workflow:

1. Load the movie review dataset.
2. Explore the data and check for missing values and sentiment distribution.
3. Preprocess review text by converting it to lowercase and removing HTML tags and non-alphabetic characters.
4. Tokenize the reviews and build a word vocabulary.
5. Convert text into sequences of token IDs.
6. Pad sequences to a fixed length of 500.
7. Split the data into training and testing sets.
8. Build and train an LSTM-based sentiment classification model.

## Model Architecture

The notebook builds the following model:

```text
Input Review (token IDs)
        |
        v
Embedding Layer (100 dimensions)
        |
        v
LSTM Layer (128 units)
        |
        v
Dense Layer (1 unit, sigmoid activation)
        |
        v
Sentiment: Positive or Negative
```

- **Embedding:** Learns a dense vector representation for each token.
- **LSTM:** Learns sequential patterns and context in the review.
- **Dense + Sigmoid:** Produces a score for binary sentiment classification.

The model is compiled with the **Adam optimizer** and **binary cross-entropy loss**.

## Dataset

The project uses the IMDB movie reviews dataset, commonly distributed as `IMDB Dataset.csv`, with two columns:

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | Review label: `positive` or `negative` |

Download the dataset from Kaggle:

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

After downloading, place `IMDB Dataset.csv` in the location expected by the notebook, or update the CSV path in the data-loading cell.

## Requirements

- Python 3
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- scikit-learn
- TensorFlow

Install the Python packages with:

```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/imdb-movie-review-sentiment-analysis-lstm.git
   ```

2. Open the project folder and download `IMDB Dataset.csv` using the dataset link above.

3. Update the dataset path in the notebook if needed. The current notebook uses:

   ```python
   df = pd.read_csv("/content/IMDB_Dataset.csv")
   ```

   For a local setup, for example, you could use:

   ```python
   df = pd.read_csv("IMDB Dataset.csv")
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `LSTM_Project.ipynb` and run the cells in order.

## Current Project Scope

This is a basic learning project focused on understanding the NLP preprocessing pipeline and building an LSTM model for binary sentiment classification.

The notebook currently trains the model for **1 epoch** with a batch size of **64**. It does not yet include a final test-set evaluation, accuracy report, confusion matrix, or prediction examples. These can be added as future improvements.

## Future Improvements

- Add model evaluation on the test dataset.
- Display accuracy, precision, recall, and F1-score.
- Add a confusion matrix.
- Try different sequence lengths, embedding dimensions, batch sizes, and epoch counts.
- Compare LSTM with GRU and Bidirectional LSTM.
- Add a function to predict sentiment for a new user-provided review.
- Save and load the trained model.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- NLP
- LSTM Neural Networks

---

**Learning project:** Built to practise text preprocessing, tokenization, sequence padding, embeddings, and LSTM-based sentiment classification.
