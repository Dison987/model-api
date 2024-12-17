from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import firebase_admin
from firebase_admin import credentials, auth
import json

# Load Firebase credentials from environment variables
firebase_credentials = os.environ.get("FIREBASE_CREDENTIALS")
if firebase_credentials:
    cred = credentials.Certificate(json.loads(firebase_credentials))
    firebase_admin.initialize_app(cred)
else:
    raise ValueError("FIREBASE_CREDENTIALS environment variable is missing")

app = Flask(__name__)

# Global conversation history
conversation_history = ""

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
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
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

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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

        device = model.device
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
