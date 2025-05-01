import asyncio
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from tenacity import retry, stop_after_attempt, wait_fixed
from responses import *
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from generate import generate
import numpy as np
import spacy
import subprocess
import sys
# Ensure the model is installed
def install_spacy_model():
    try:
        # Try to load the model
        spacy.load('en_core_web_sm')
    except IOError:
        # If the model is not found, download and install it
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Call the function to ensure the model is available
install_spacy_model()

# Now you can load the model and continue with the rest of your bot logic
nlp = spacy.load('en_core_web_sm')

from polariser import *
from emotion_model import detect_emotion
import spacy



TOKEN = "7983866766:AAE4_35JjKCbPzzyRCHwce1QbV1aoHfHYks"  # Replace with your actual token from BotFather

data = pd.read_csv('pre_data (1).csv')     #loading the preprocessed data
X = data['text']
y = data['category']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

svm_file=open('svm_model_probablity','rb')
svm_model=pickle.load(svm_file)
svm_file.close()

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)


async def start(update: Update, context):
    user = update.message.from_user
    user_id = str(user.id)
    context.user_data[user_id] = {
        "values": np.array([0.0] * 7),
        "length": 0, 
        "frequency":{
            "Anxiety":0,
            "Normal":0,
            "Depression":0,
            "Suicidal":0,
            "Stress":0,
            "Bipolar":0,
            "Personality disorder":0
        }
    }
    await update.message.reply_text("Hello! Welcome to the bot.")
    await update.message.reply_text("How can I help you?")


async def report(update: Update, context):
    user = update.message.from_user
    username = user.username or user.first_name or str(user.id)
    first_name = user.first_name or "User"
    user_id = str(user.id)

    # Safe check: if user_id data does not exist
    if user_id not in context.user_data:
        await update.message.reply_text("Please start the bot and chat first before requesting a report.")
        return

    values = np.array(list(context.user_data[user_id]["frequency"].values()))
    total = sum(values)

    if total == 0:
        await update.message.reply_text("Report could not be generated due to lack of conversations.")
        return    

    avg_values = values / total

    generate(first_name, username, avg_values)

    pdf_path = f"PDF/{username}.pdf"

    if os.path.exists(pdf_path):
        await update.message.reply_document(document=open(pdf_path, 'rb'))
        await update.message.reply_text("✅ Your mental health report is ready!")
    else:
        await update.message.reply_text("❌ Report generation failed. Please try again.")



async def echo(update: Update, context):
    user = update.message.from_user
    user_id = str(user.id)
    user_input = update.message.text
    
    if user_input.lower() in "hello hi hey what's up? howdy  greetings welcome hiya yo  to see you how's it going? nice to meet you":
        await update.message.reply_text("Hello there !")
        return
    
    # Analyze the user's input sentiment
    sentiment = get_polarity(user_input)
    new_text = [user_input]
   
    
    new_text_vectorized = vectorizer.transform(new_text)
    category_probabilities = svm_model.predict_proba(new_text_vectorized)

    svm_prediction = svm_model.predict(new_text_vectorized)
    svm_predicted_label = label_encoder.inverse_transform(svm_prediction)[0]

    # BERT Emotion Detection
    emotion_label, emotion_probs = detect_emotion(user_input)

    # Combine logic (example: if negative sentiment + sadness/fear emotion)
    if sentiment < -0.3 and emotion_label in ["sadness", "fear"]:
        response = "You seem really low. Want to talk more about it or should I recommend something helpful?"
    elif sentiment > 0.3 and emotion_label in ["joy", "surprise"]:
        response = "That’s awesome! Want to share more?"
    else:
        response = get_response(sentiment)  # Fallback to basic polarity-based response

    # Update session data
    context.user_data[user_id]["values"] += category_probabilities[0]
    context.user_data[user_id]["length"] += 1
    context.user_data[user_id]['frequency'][svm_predicted_label] += 1

    # Reply with BERT emotion + SVM category + sentiment-based message
    await update.message.reply_text(f"Detected Emotion: {emotion_label}\nCategory: {svm_predicted_label}")
    await update.message.reply_text(response)


   

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def run_bot(application):
    application.run_polling()

def main():

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("report", report))
    application.add_handler(MessageHandler(filters.TEXT, echo))
    
    print("Bot is running...")
    asyncio.run(application.run_polling())
    run_bot(application)

if __name__ == "__main__":
    main()
