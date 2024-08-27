import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from transformers import BertTokenizer, BertForTokenClassification
from typing import Dict

app = FastAPI()

# Load the OCR model (replace with your model path)
def load_model():
    model = tf.keras.models.load_model('path_to_your_model')
    return model

# Preprocess the image for the model
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Resize image to the size expected by the model (example size: 128x128)
    resized_image = cv2.resize(thresh, (128, 128))
    # Normalize the image
    normalized_image = resized_image / 255.0
    # Add batch dimension
    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image

# Extract text from image using the model
def extract_text_from_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    # Post-process predictions to extract text (this depends on your model)
    # For demonstration, assume predictions are text logits
    predicted_text = np.argmax(predictions, axis=-1)
    return predicted_text

# NLP Processing with BERT
def nlp_processing(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # Extract entities (implement custom logic here)
    # Example: get predicted labels
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=2)
    return predictions

@app.post('/extract_details')
async def extract_details(file: UploadFile = File(...)):
    print("request "+str(file))
    if file:
        print(f"Received file: {file.filename}")
        image = await file.read()
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        model = load_model()
        text = extract_text_from_image(image, model)
        details = nlp_processing(text)
        return {"details": details}
    else:
        return {"error": "No file uploaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
