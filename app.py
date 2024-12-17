# Support resetChat()

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import firebase_admin
from firebase_admin import credentials, auth

# Initialize the SDK
cred = credentials.Certificate("C:/Users/PC/Desktop/FYP/fyp-db-4e0f7-firebase-adminsdk-ihmo0-a1f109394e.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

# Global conversation history
conversation_history = ""

def reset_chat():
    """Resets the global conversation history"""
    global conversation_history
    conversation_history = ""

def load_model():
    try:
        global model, tokenizer  # Make these global so they can be used elsewhere
        model = AutoModelForCausalLM.from_pretrained(
            './mistralv2_saved_model_bigv2',  
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare the model for LoRA fine-tuning if applicable
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA configuration 
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
        tokenizer = AutoTokenizer.from_pretrained(
            './mistralv2_saved_model_tokenized_bigv2',
        )

        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False

def generate_response(user_input):
    """Generate response using the loaded model, maintaining conversation history"""
    global conversation_history

    try:
        # Append user input to the conversation history
        conversation_history += f"User: {user_input}\nChatbot:"

        # Tokenize the conversation history
        inputs = tokenizer(
            conversation_history,
            return_tensors="pt",
            truncation=True,  # Truncate input if it exceeds max length
            max_length=1024,  # Adjust max length to accommodate context
            padding=True      # Pad input for batch compatibility
        )

        # Move inputs to the model's device
        device = model.device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate a response from the model
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,          # Limit response length
            do_sample=True,              # Enable sampling for variety
            temperature=0.9,             # Adjust creativity
            top_k=50,                    # Limit sampling to top 50 tokens
            top_p=0.95,                  # Use nucleus sampling
            repetition_penalty=1.2,      # Penalize repeated phrases
            pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad token ID
        )

        # Decode the generated response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the chatbot's response from the generated text
        chatbot_response = response.split("Chatbot:")[-1].strip()

        # Append the chatbot's response to the conversation history
        conversation_history += f" {chatbot_response}\n"

        # Truncate conversation history if it exceeds a certain length (e.g., 1024 tokens)
        max_history_length = 1024
        if len(conversation_history) > max_history_length:
            conversation_history = conversation_history[-max_history_length:]

        return chatbot_response

    except Exception as e:
        return f"Error generating response: {str(e)}"


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat requests"""
    try:
        # Get question and optional reset flag from the request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question in request'
            }), 400

        question = data['question']
        
        # Check for reset flag
        if data.get('reset', False):  # If 'reset' is present and True
            reset_chat()

        # Generate response
        response = generate_response(question)

        return jsonify({
            'response': response
        })
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Endpoint to reset the chat history"""
    try:
        reset_chat()
        return jsonify({
            'message': 'Conversation history reset successfully.'
        })
    except Exception as e:
        return jsonify({
            'error': f'Error resetting chat: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

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
    # Load model when starting the server
    if load_model():
        # Run the Flask app
        # For production, you might want to use a proper WSGI server like gunicorn
        app.run(host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting.")
