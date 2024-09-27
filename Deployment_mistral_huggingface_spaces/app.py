import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
import re

# Load model and tokenizer from Hugging Face Hub
model_name = "Rsood/mistral-instruct-v2-phishing-detection"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # Automatically handle device placement (GPU/CPU)
    torch_dtype=torch.float16,     # Use half-precision to save memory (effective on GPU)
    low_cpu_mem_usage=True         # Reduces CPU memory usage during model loading
)

# Enable gradient checkpointing for memory efficiency during inference
model.gradient_checkpointing_enable()
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = model.config.eos_token_id

# Preprocessing functions
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'http\S+|www\S+|https\S+', 'URLfound', text, flags=re.MULTILINE)
    text = text.lower()
    return text

def truncate_content(content, max_length=600): 
    return content[:max_length] + "..." if len(content) > max_length else content

def standardize_output(result):
    result = result.strip()
    if '1' in result:
        return "Phishing"
    elif '0' in result:
        return "Non-Phishing"
    else:
        return "You Broke the Model. I curse you!"  # Invalid output

# Function to classify the email
def classify_email(email_text):
    # Preprocess the email content
    clean_email = clean_text(email_text)
    truncated_email = truncate_content(clean_email)

    # Prepare the prompt
    prompt_template = """
    [INST] You are a phishing detection classifier. Classify the email as phishing (1) or non-phishing (0). Classify only phishing emails as phishing (1) and spam and non-phishing as non-phishing (0).
    Return ONLY the integer 1 or 0. Do not provide any explanation or additional text.
    ### Example 1:
    Email Content:
    "Hello"
    Your Response: 0
    ### Example 2:
    Email Content:
    "Your account has been hacked click the link below URLFOUND"
    Your Response: 1
    ### Now classify this email:
    Email Content:
    "{content}"
    Your Response: [/INST]
    """
    prompt = prompt_template.format(content=truncated_email)

    # Tokenize the prompt
    max_seq_length = 1024  # Assuming this is the max length used during training
    sample_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length
    ).to("cuda")

    # Generate the output
    with torch.no_grad():
        sample_output = model.generate(
            **sample_input,
            max_new_tokens=2,  # Limit the output length
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the output
    sample_prediction = tokenizer.decode(
        sample_output[0, sample_input['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return standardize_output(sample_prediction)

# Gradio Interface
interface = gr.Interface(
    fn=classify_email,
    inputs="text",  # Single text input
    outputs="text",  # Text output
    title="Phishing Email Detection",
    description="Enter email content and the model will classify it as Phishing (1) or Non-Phishing (0)."
)

# Launch the Gradio interface
interface.launch(share=True)
#test
