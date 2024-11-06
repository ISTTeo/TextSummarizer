from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
def generate_text(
	model,
	tokenizer,
	messages,
	generation_args):
	    
	pipe = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
        #device=0
	)
	    
	output = pipe(messages, **generation_args)
	    
	return output[0]['generated_text']
	    
	    
def run_phi3(system_command, user_command, model, tokenizer, generation_args):
	messages = [{"role": "system", "content": system_command},
		        {"role": "user", "content": user_command}]

	return generate_text(model, tokenizer, messages, generation_args)

def model_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Tokenizer initialization
    
    model = AutoModelForCausalLM.from_pretrained(
    	model_id, 
    	torch_dtype="auto", 
    	trust_remote_code=True,
        quantization_config=bnb_config
	)
    	
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def get_token_count(tokenizer, text):
    tokens = tokenizer.encode(text)
    return len(tokens)



"""
import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

model_id = "microsoft/Phi-3.5-mini-instruct"
# BitsAndBytesConfig for 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use float16 for 8-bit quantization
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
"""