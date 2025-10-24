# Usa a imagem oficial do Ollama
FROM ollama/ollama

# Baixa o modelo LLaMA 3 (pode escolher 7B ou 8B)
RUN ollama pull llama3

# Expõe a porta padrão
EXPOSE 11434

# Inicia o servidor do Ollama
CMD ["serve"]
