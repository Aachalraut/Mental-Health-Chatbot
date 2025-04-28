from datasets import load_dataset

# Load the GoEmotions dataset from Hugging Face
dataset = load_dataset("go_emotions")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model from Hugging Face
model_name = "bert-base-uncased"  # BERT pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=27)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from transformers import TrainingArguments

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",              # Output directory for saving model
    evaluation_strategy="epoch",         # Evaluation after each epoch
    learning_rate=2e-5,                  # Learning rate
    per_device_train_batch_size=16,      # Batch size for training
    per_device_eval_batch_size=64,       # Batch size for evaluation
    num_train_epochs=3,                  # Number of epochs to train
    weight_decay=0.01,                   # Weight decay for regularization
    logging_dir="./logs",                # Directory for logs
    logging_steps=10,                    # Logging frequency
)

from transformers import Trainer

# Initialize the Trainer
trainer = Trainer(
    model=model,                          # The model to train
    args=training_args,                   # Training arguments
    train_dataset=tokenized_datasets["train"],   # Training dataset
    eval_dataset=tokenized_datasets["validation"],  # Validation dataset
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert_emotion")

# Save the tokenizer
tokenizer.save_pretrained("./fine_tuned_bert_emotion")

from transformers import pipeline

# Load the fine-tuned model
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test with a user input
user_input = "I'm feeling so happy today!"
emotion = emotion_classifier(user_input)

print(f"Predicted Emotion: {emotion[0]['label']}, Confidence: {emotion[0]['score']}")
