from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Keep conversation history
chat_history = ""

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    # Add the user message to the conversation
    chat_history += f"User: {user_input}\nAssistant:"

    # Tokenize with history
    inputs = tokenizer(chat_history, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

    # Decode and extract assistant's reply
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only new assistant reply
    assistant_reply = response.split("Assistant:")[-1].strip()
    print(f"Assistant: {assistant_reply}")

    # Append reply to chat history
    chat_history += f" {assistant_reply}\n"
