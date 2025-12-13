# main.py
import hashlib
import json
import time
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


# --- nlp_detector.py ---
class NLPDetector:
    def __init__(self):
        """
        Initializes the NLPDetector with necessary tools and a simple pre-trained model.
        In a real application, a much larger and more diverse dataset would be used.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        self._train_model() # Train the model upon initialization

    def _preprocess_text(self, text):
        """
        Cleans and preprocesses the input text.
        Steps: lowercase, remove punctuation, remove non-alphabetic characters,
        remove stopwords, and lemmatize.
        """
        text = text.lower()  # Convert to lowercase
        # Keep only alphabetic characters and spaces
        cleaned_text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        words = cleaned_text.split()  # Tokenize
        # Remove stopwords and lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def _train_model(self):
        """
        Trains a simple Logistic Regression model for fake news detection.
        This uses a very small, hardcoded dataset for demonstration.
        In a real-world scenario, you would load a large dataset (e.g., from CSV)
        and use more sophisticated models.
        """
        # Greatly Expanded hardcoded sample data for demonstration purposes
        # '0' for real, '1' for fake. More diverse examples added for better demonstration.
        sample_news = [
            # Real News Examples
            ("Local city council approves new budget for park improvements, aiming to enhance public spaces by next fiscal year.", "0"),
            ("Scientists confirm discovery of new exoplanet in distant galaxy, showing potential for further research into alien life.", "0"),
            ("Community leaders discuss plans for upcoming annual charity event, expecting record participation this year.", "0"),
            ("New research suggests moderate coffee consumption may reduce certain health risks, according to a peer-reviewed study.", "0"),
            ("Weather forecast predicts sunny skies and warm temperatures for the entire weekend, ideal for outdoor activities.", "0"),
            ("School board announces new curriculum changes for next academic year, focusing on STEM education from elementary levels.", "0"),
            ("Tech company releases innovative new smartphone with advanced features, setting a new benchmark for mobile technology.", "0"),
            ("Archaeologists uncover ancient ruins in remote desert region, hinting at a previously unknown civilization.", "0"),
            ("World leaders gather for climate summit to discuss environmental policies, seeking global cooperation on climate change.", "0"),
            ("Local police report significant decrease in crime rates this quarter, attributing it to new community programs.", "0"),
            ("New art exhibition opens at downtown gallery, featuring works from talented local and emerging artists.", "0"),
            ("University study shows profound benefits of regular sleep on cognitive function and overall mental well-being.", "0"),
            ("Economists predict steady growth in the national GDP for the upcoming fiscal quarter, boosting investor confidence.", "0"),
            ("Medical breakthrough: new drug shows promise in treating rare genetic disorder in clinical trials.", "0"),
            ("Sports team wins championship after thrilling overtime victory, celebrating with fans in a city-wide parade.", "0"),
            ("Government agency issues new guidelines for cybersecurity, advising citizens on safe online practices.", "0"),
            ("Environmental group launches campaign to clean up local rivers, encouraging public participation.", "0"),
            ("International space station completes complex docking procedure, resupplying astronauts for upcoming missions.", "0"),
            ("Philanthropist donates large sum to educational programs, aiming to support underprivileged students.", "0"),
            ("New public transportation route launched, connecting previously underserved areas of the city.", "0"),

            # Fake News Examples
            ("Breaking news: Giant sentient potatoes declare war on humanity, demanding world domination and free ketchup!", "1"),
            ("Urgent warning: Eating clouds after midnight causes spontaneous levitation and makes you glow green!", "1"),
            ("Government reveals aliens have been living among us for centuries, disguised as common house plants and secretly controlling wifi signals.", "1"),
            ("Miracle cure discovered: Breathing underwater daily reverses aging, cures all diseases, and grants superpowers!", "1"),
            ("Famous movie star arrested for attempting to replace the moon with a giant disco ball, causing tidal chaos.", "1"),
            ("A secret society controls the world's gravity using ancient magic crystals, making objects float randomly.", "1"),
            ("Scientists prove that dragons are real and guard a treasure hidden in active volcanoes, accessible only by singing opera.", "1"),
            ("Your toaster is secretly a supercomputer plotting to take over your home, starting with controlling your breakfast.", "1"),
            ("Global warming is a myth, definitively proven by a single snowflake found in July in the Sahara Desert.", "1"),
            ("Doctors discover new vaccine that makes you invisible for a day, but only when you hum the national anthem.", "1"),
            ("Rare blue moon will turn the sky purple tomorrow night, causing all animals to speak Latin.", "1"),
            ("Study finds that talking to plants helps them pay your bills and do your laundry.", "1"),
            ("New fashion trend: wearing socks on your hands is the latest craze, making typing impossible but looking cool.", "1"),
            ("Elvis Presley spotted on Mars, building a casino with alien friends.", "1"),
            ("World's oceans turn to jelly overnight, trapping all ships and allowing people to walk on water.", "1"),
            ("Time travel confirmed! Local man claims to have seen dinosaurs riding bicycles in the future.", "1"),
            ("The sun is actually a giant lightbulb run by squirrels, according to a leaked document.", "1"),
            ("Scientists invent a pillow that teaches you quantum physics while you sleep.", "1"),
            ("All cats gain the ability to sing opera and are forming a global choir.", "1"),
            ("New law requires everyone to wear a banana peel on their head for good luck.", "1")
        ]

        texts = [item[0] for item in sample_news]
        labels = [int(item[1]) for item in sample_news]

        # Preprocess all texts for training
        preprocessed_texts = [self._preprocess_text(text) for text in texts]

        # Initialize TF-IDF Vectorizer
        # Adjusted max_features for the slightly larger dataset
        self.vectorizer = TfidfVectorizer(max_features=3000) # Increased max_features further
        # Fit vectorizer and transform training data
        X = self.vectorizer.fit_transform(preprocessed_texts)
        y = labels

        # Train a Logistic Regression model
        self.model = LogisticRegression(max_iter=2000, solver='liblinear') # Increased max_iter and specified solver for robustness
        self.model.fit(X, y)
        print("NLP Detector Model trained successfully with a highly diversified sample dataset.")

    def predict_fake_news(self, news_text):
        """
        Predicts whether a given news text is fake or real, along with a confidence score.
        Returns a tuple: (classification_string, confidence_percentage_string)
        """
        # Preprocess the input text
        preprocessed_text = self._preprocess_text(news_text)

        # Check if the preprocessed text is empty or very short, implying non-linguistic input
        # This helps in handling inputs like "38-4=32" more gracefully.
        if not preprocessed_text or len(preprocessed_text.split()) < 2:
            return "Cannot Classify (Non-Textual/Irrelevant Input)", "N/A"

        # Transform the single text using the *fitted* vectorizer
        text_vector = self.vectorizer.transform([preprocessed_text])

        # Get prediction (0 or 1) and probabilities
        prediction = self.model.predict(text_vector)
        # Probabilities for [class 0 (real), class 1 (fake)]
        probabilities = self.model.predict_proba(text_vector)[0]

        # Determine the classification string and confidence
        if prediction[0] == 1:
            classification = "Fake"
            confidence = probabilities[1] * 100 # Confidence for 'fake' class
        else:
            classification = "Real"
            confidence = probabilities[0] * 100 # Confidence for 'real' class

        return classification, f"{confidence:.2f}%"

# --- blockchain.py ---
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        """
        Represents a single block in the blockchain.
        Args:
            index (int): The position of the block in the chain.
            timestamp (str): The time the block was created.
            data (dict): The data stored in the block (e.g., news content, classification).
            previous_hash (str): The hash of the previous block in the chain.
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self._calculate_hash() # Calculate the hash of this block

    def _calculate_hash(self):
        """
        Calculates the SHA-256 hash for the block.
        The hash is computed based on all the block's properties.
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        """
        Initializes the blockchain by creating the genesis block.
        """
        self.chain = [self._create_genesis_block()]

    def _create_genesis_block(self):
        """
        Creates the very first block in the blockchain (the genesis block).
        """
        return Block(0, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), {"message": "Genesis Block"}, "0")

    def get_latest_block(self):
        """
        Returns the most recently added block in the chain.
        """
        return self.chain[-1]

    def add_block(self, data):
        """
        Adds a new block to the blockchain.
        Args:
            data (dict): The data to be stored in the new block.
        Returns:
            Block: The newly created block.
        """
        previous_block = self.get_latest_block()
        new_block = Block(
            index=previous_block.index + 1,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data=data,
            previous_hash=previous_block.hash
        )
        self.chain.append(new_block)
        print(f"Block #{new_block.index} added to the blockchain.")
        return new_block

    def is_chain_valid(self):
        """
        Checks if the entire blockchain is valid by verifying hashes.
        Returns:
            bool: True if the chain is valid, False otherwise.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Check if the current block's hash is correctly calculated
            if current_block.hash != current_block._calculate_hash():
                print(f"Block #{current_block.index}: Current hash mismatch!")
                return False

            # Check if the current block's previous_hash points to the actual previous block's hash
            if current_block.previous_hash != previous_block.hash:
                print(f"Block #{current_block.index}: Previous hash mismatch!")
                return False
        return True

    def display_chain(self):
        """
        Prints the details of all blocks in the blockchain.
        """
        print("\n--- Blockchain ---")
        if not self.chain:
            print("Blockchain is empty.")
            return

        for block in self.chain:
            print(f"Block #{block.index}")
            print(f"  Timestamp: {block.timestamp}")
            print(f"  Data: {block.data}")
            print(f"  Previous Hash: {block.previous_hash}")
            print(f"  Current Hash: {block.hash}")
            print("-" * 30)
        print("------------------\n")


# --- main.py (main application logic) ---
def main():
    """
    Main function to run the Fake News Detection System.
    Provides a command-line interface for user interaction.
    """
    nlp_detector = NLPDetector()
    blockchain = Blockchain()

    print("Welcome to the Blockchain and NLP Fake News Detection System!")
    print("----------------------------------------------------------")

    while True:
        print("\nChoose an option:")
        print("1. Analyze new news article")
        print("2. View blockchain")
        print("3. Check blockchain validity")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            news_text = input("\nEnter the news article text to analyze: \n")
            if not news_text.strip():
                print("News text cannot be empty. Please try again.")
                continue

            print("\nAnalyzing news...")
            # Predict if the news is fake or real using NLP and get confidence
            prediction, confidence = nlp_detector.predict_fake_news(news_text)

            print(f"\n--- Analysis Result ---")
            print(f"The system has analyzed the news article provided.")

            if "Cannot Classify" in prediction:
                print(f"The input '{news_text}' appears to be non-textual or irrelevant for news analysis.")
                print(f"This NLP model is designed for natural language (sentences, paragraphs) and cannot meaningfully classify such data.")
            else:
                print(f"Based on linguistic patterns and learned characteristics,")
                print(f"the NLP model predicts this news is likely: {prediction} (Confidence: {confidence}).")
                # Prepare data to be stored in the blockchain ONLY if it's a valid classification
                block_data = {
                    "news_content_hash": hashlib.sha256(news_text.encode()).hexdigest(), # Store hash of content
                    "nlp_prediction": prediction,
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                # Add the prediction and news hash to the blockchain
                blockchain.add_block(block_data)
                print("News analysis results recorded immutably on the blockchain.")

            print(f"-----------------------")


        elif choice == '2':
            blockchain.display_chain()

        elif choice == '3':
            print("\nChecking blockchain validity...")
            if blockchain.is_chain_valid():
                print("Blockchain is valid! No tampering detected, ensuring data integrity.")
            else:
                print("Blockchain is INVALID! Tampering detected, indicating a potential compromise of records.")

        elif choice == '4':
            print("Exiting system. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
