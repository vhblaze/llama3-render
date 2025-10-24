FROM ollama/ollama

# Expõe a porta padrão
EXPOSE 11434

# Inicia o Ollama e faz o download do modelo automaticamente
CMD bash -c "ollama serve & sleep 5 && ollama pull llama3 && tail -f /dev/null"
