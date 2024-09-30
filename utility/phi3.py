from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_text(
	model,
	tokenizer,
	messages,
	generation_args):
	    
	pipe = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
	)
	    
	output = pipe(messages, **generation_args)
	    
	return output[0]['generated_text']
	    
	    
def run_phi3(system_command, user_command, model, tokenizer, generation_args):
	messages = [{"role": "system", "content": system_command},
		        {"role": "user", "content": user_command}]

	return generate_text(model, tokenizer, messages, generation_args)

def model_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
    	model_id, 
    	torch_dtype="auto", 
    	trust_remote_code=True, 
	)
    model.to(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def get_token_count(tokenizer, text):
    tokens = tokenizer.encode(text)
    return len(tokens)