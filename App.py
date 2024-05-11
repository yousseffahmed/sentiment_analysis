from flask import Flask, render_template, request, jsonify
from googletrans import Translator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
matplotlib.use('Agg')
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sia = SentimentIntensityAnalyzer()

def translate_arabic_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='ar', dest='en')
    return translated_text.text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/english.html')
def english():
    return render_template('english.html')

@app.route('/arabic.html')
def arabic():
    return render_template('arabic.html')

@app.route('/csv.html')
def csv():
    return render_template('csv.html')

@app.route('/analysis.html')
def analysis():
    return render_template('analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']

        sentiment = sia.polarity_scores(text)
        
        # Prepare response
        response = {
            'text': text,
            'sentiment': sentiment
        }
        
        # Return JSON response
        return jsonify(response)
    
@app.route('/analyzeArabic', methods=['POST'])
def analyzeArabic():
    if request.method == 'POST':
        text = request.form['text']

        sentiment = sia.polarity_scores(translate_arabic_to_english(text))
        
        # Prepare response
        response = {
            'text': text,
            'sentiment' : sentiment
        }
        
        return jsonify(response)
    
@app.route('/analyzeCSV', methods=['POST'])
def analyzeCSV():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    # Read CSV file
    df = pd.read_csv(file)
    df = df.head(500)

    # Perform sentiment analysis
    res = {}
    for i, row in df.iterrows():
        text = row['Text']
        myId = row['Id']
        res[myId] = sia.polarity_scores(text)

    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'Id'})
    vaders = vaders.merge(df, how='left')

    # Calculate sentiment scores
    vaders['sentiment'] = vaders['compound'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

    # Aggregate sentiment scores
    sentiment_aggregated = vaders['sentiment'].value_counts()

    # Plot Number of Positive and Negative Reviews
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sentiment_aggregated.plot(kind='bar', color=['green', 'red', 'black'], ax=ax1)
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Number of Positive and Negative Reviews')

    # Plot Distribution of Sentiment Scores
    sentiment_scores = [sia.polarity_scores(text)['compound'] for text in df['Text']]
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(sentiment_scores, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution of Sentiment Scores')
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    # Plot Sentiment Scores of Reviews
    vaders['sentiment_score'] = vaders['Text'].apply(lambda text: sia.polarity_scores(text)['compound'])
    positive_reviews = vaders[vaders['sentiment_score'] > 0]
    negative_reviews = vaders[vaders['sentiment_score'] < 0]
    positive_reviews['Sentiment'] = 'Positive'
    negative_reviews['Sentiment'] = 'Negative'
    combined_reviews = pd.concat([positive_reviews, negative_reviews])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.swarmplot(data=combined_reviews, x='Sentiment', y='sentiment_score', palette={'Positive': 'green', 'Negative': 'red'})
    ax3.set_title('Sentiment Scores of Reviews')
    ax3.set_xlabel('Sentiment')
    ax3.set_ylabel('Sentiment Score')

    # Prepare ground truth sentiments
    ground_truth = df['sentiment']

    # Convert sentiments to numerical labels
    sentiment_labels = {'positive': 0, 'negative': 1, 'neutral': 2}
    ground_truth_numeric = ground_truth.map(sentiment_labels)
    predicted_sentiments = vaders['sentiment'].map(sentiment_labels)

    # Add predicted sentiments to the DataFrame
    vaders['sentiment_pred'] = predicted_sentiments

    # Compute confusion matrix
    cm = confusion_matrix(ground_truth_numeric, predicted_sentiments)

    # Visualize confusion matrix
    classes = ['positive', 'negative', 'neutral']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Get misclassified indices
    misclassified_indices = np.where(ground_truth_numeric != predicted_sentiments)[0]

    # Filter out misclassified rows
    misclassified = vaders.iloc[misclassified_indices]

    # Output the mispredicted texts along with their predicted and actual labels
    misclassified_texts = misclassified[['Text', 'pred', 'sentiment_pred']]
    print("Misclassified Texts:")
    print(misclassified_texts)
    
    # Convert confusion matrix plot to base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confusion_matrix_plot = base64.b64encode(buf.getvalue()).decode()
    plt.close()  # Close the plot to avoid overlapping with other plots

    # Convert other plots to base64 encoded strings
    plots = []
    for fig in [fig1, fig2, fig3]:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plots.append(base64.b64encode(buf.getvalue()).decode())

    return render_template('analysis.html', confusion_matrix_plot=confusion_matrix_plot, plots=plots, misclassified_texts=misclassified_texts)


if __name__ == '__main__':
    app.run(debug=True)
