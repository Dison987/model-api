from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import firebase_admin
from firebase_admin import credentials, auth
import json

# Path to Firebase credentials JSON file
firebase_credentials_path = './fyp-db-4e0f7-firebase-adminsdk-ihmo0-a1f109394e.json'

# Load Firebase credentials from JSON file
if os.path.exists(firebase_credentials_path):
    cred = credentials.Certificate(firebase_credentials_path)
    firebase_admin.initialize_app(cred)
else:
    raise ValueError(f"Firebase credentials file not found at {firebase_credentials_path}")

app = Flask(__name__)

# Global conversation history
conversation_history = ""

# Hugging Face token
hf_token = "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"

def reset_chat():
    """Resets the global conversation history"""
    global conversation_history
    conversation_history = ""

def load_model():
    try:
        global model, tokenizer
        # Model path (change to relative paths or environment variable)
        model_path = os.environ.get("MODEL_PATH", "./mistralv2_saved_model_bigv2")
        tokenizer_path = os.environ.get("TOKENIZER_PATH", "./mistralv2_saved_model_tokenized_bigv2")

        # Use 'cpu' explicitly when loading the model for CPU support
        device = torch.device("cpu")

        # Load the model with authentication token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_auth_token=hf_token,  # Pass the Hugging Face token here
            load_in_4bit=False,  # Disable 4-bit loading for CPU
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Set to None to avoid GPU-specific setup
        )

        # Prepare the model for LoRA fine-tuning
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        # Load the tokenizer with authentication token
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)  # Pass the token here

        # Move model to CPU
        model.to(device)

        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False

def generate_response(user_input):
    global conversation_history
    try:
        conversation_history += f"User: {user_input}\nChatbot:"

        inputs = tokenizer(
            conversation_history,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )

        # Ensure inputs and model are on the same device (CPU)
        device = torch.device("cpu")  # Use CPU explicitly
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chatbot_response = response.split("Chatbot:")[-1].strip()
        conversation_history += f" {chatbot_response}\n"

        max_history_length = 1024
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[-max_history_length:]

        return chatbot_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request'}), 400

        question = data['question']
        if data.get('reset', False):
            reset_chat()

        response = generate_response(question)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        reset_chat()
        return jsonify({'message': 'Conversation history reset successfully.'})
    except Exception as e:
        return jsonify({'error': f'Error resetting chat: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route("/api/delete-user", methods=["POST"])
def delete_user():
    try:
        data = request.json
        uid = data.get("uid")
        auth.delete_user(uid)
        return jsonify({"message": f"User {uid} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    if load_model():
        port = int(os.environ.get('PORT', 5000))  # Use Railway-assigned port
        app.run(host='0.0.0.0', port=port)
    else:
        print("Failed to load model. Exiting.")
