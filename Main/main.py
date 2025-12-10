from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# add special token <CONFIDENCE> to the tokenizer
num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["<CONFIDENCE>"]})
print(f"Added {num_added_tokens} special token(s) to the tokenizer")
print(f"Original vocab size: {len(tokenizer) - num_added_tokens}")
print(f"New vocab size: {len(tokenizer)}")

# resize model embeddings to match new tokenizer vocab size
model.resize_token_embeddings(len(tokenizer))
print(f"Model embeddings resized to: {model.get_input_embeddings().weight.shape[0]}")

# verify the token was added
confidence_token_id = tokenizer.convert_tokens_to_ids("<CONFIDENCE>")
print(f"<CONFIDENCE> token ID: {confidence_token_id}")


q1 = "A tank has two inlet pipes and one outlet pipe. Inlet A fills the tank in 3 hours. Inlet B fills the tank in 4 hours. Outlet C empties the tank in 6 hours. If all three pipes are opened at the same time when the tank is empty, how long will it take to fill the tank?"


# prepare the model input
prompt = q1
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)







