#!/bin/bash
# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready (timeout after 120 seconds)
echo "Waiting for Ollama server to start..."
TIMEOUT=120
ELAPSED=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Ollama server failed to start after ${TIMEOUT}s"
        exit 1
    fi
done
echo "Ollama server is ready (took ~${ELAPSED}s)."

# Create the model if it doesn't already exist
if ! ollama list | grep -q "tamil-llama"; then
    echo "Creating tamil-llama model from GGUF..."
    if ollama create tamil-llama -f /models/tamil-llama/Modelfile; then
        echo "tamil-llama model created successfully."
    else
        echo "ERROR: Failed to create tamil-llama model"
        exit 1
    fi
else
    echo "tamil-llama model already exists."
fi

# Handle graceful shutdown
trap "kill $OLLAMA_PID; exit 0" SIGTERM SIGINT

# Keep the server running in the foreground
wait $OLLAMA_PID
