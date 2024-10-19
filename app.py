# app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("/Applications/Arun/AI/AAI-520/Project/fine_tuned_dialogpt_medium", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/Applications/Arun/AI/AAI-520/Project/fine_tuned_dialogpt_medium", local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Constants
MAX_HISTORY_TURNS = 5
MAX_HISTORY_TOKENS = 512

def generate_response(prompt, conversation_history, max_length=50):
    full_prompt = construct_prompt(conversation_history, prompt)
    
    input_ids = tokenizer.encode(full_prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    
    if input_ids.shape[1] > MAX_HISTORY_TOKENS:
        input_ids = input_ids[:, -MAX_HISTORY_TOKENS:]
        attention_mask = attention_mask[:, -MAX_HISTORY_TOKENS:]
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(full_prompt):].strip()

def construct_prompt(conversation_history, current_prompt):
    prompt_parts = [
        "The following is a conversation about movies, particularly '10 Things I Hate About You'. Respond in the style of the movie's characters:",
        *conversation_history[-MAX_HISTORY_TURNS:],
        f"Human: {current_prompt}",
        "AI:"
    ]
    return "\n".join(prompt_parts)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    conversation_history = request.json['history']
    
    response = generate_response(user_message, conversation_history)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)