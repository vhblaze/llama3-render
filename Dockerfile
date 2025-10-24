# Usa a imagem oficial do Ollama
FROM ollama/ollama

# Muda o diretório de trabalho (opcional)
WORKDIR /app

# Exponha a porta do servidor Ollama
EXPOSE 11434

# Substitui o entrypoint padrão (que é 'ollama') por 'bash'
ENTRYPOINT ["/bin/bash", "-c"]

# Comando que roda o servidor Ollama e baixa o modelo
CMD ["ollama serve & sleep 5 && ollama pull llama3:7b-q4_K_M && tail -f /dev/null"]
