import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
    device = "cuda"
else:
    print("⚠ Rodando na CPU comece a rezar pela minha memoria ram")
    device = "cpu"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model = model.to(device)

print("Modelo carregado em:", next(model.parameters()).device)

prompt = (
   """<|system|>\n"
    Você é um assistente que responde em português.\n
    A sua formatação das respostas ser no formato de saida para o console do vscode.\n    
    <|user|>\n
    Explique de forma simples o que é uma cadeia de Markov.\n
    <|assistant|>"""
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("Inputs estão em:", inputs.input_ids.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=600,
    temperature=0.7,
    do_sample=True,
)

print("\n================ RESPOSTA ================\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))