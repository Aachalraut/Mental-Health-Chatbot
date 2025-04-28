from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load your model and tokenizer (make sure you have the correct model loaded)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # Replace with your actual model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Replace with your actual tokenizer
emotion_labels = [ "anger", "disgust", "fear", "joy", "sadness", "surprise"]


def detect_emotion(user_input):
    # Tokenize the input text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model outputs (logits)
    with torch.no_grad():  # Disable gradients as we're not training
        outputs = model(**inputs)
    
    # Get the predicted class id by finding the max logit (probability)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    print(f"Predicted class id: {predicted_class_id}")  # Debug print

    # Check if the predicted_class_id is valid
    if predicted_class_id < 0 or predicted_class_id >= len(emotion_labels):
        print(f"Invalid predicted class id: {predicted_class_id}. Emotion labels size: {len(emotion_labels)}")
        return "unknown", None  # Or handle this case as needed

    return emotion_labels[predicted_class_id], None  # Return emotion label (and any other output you need)
