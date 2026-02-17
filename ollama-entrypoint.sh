#!/bin/bash

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama port to be open (no curl/wget needed)
echo "Waiting for Ollama server to start..."
TIMEOUT=60
ELAPSED=0

until timeout 1 bash -c '</dev/tcp/127.0.0.1/11434' 2>/dev/null; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Ollama server failed to start after ${TIMEOUT}s"
        exit 1
    fi
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "Still waiting... (${ELAPSED}s elapsed)"
    fi
done

echo "Ollama server is ready (took ~${ELAPSED}s)."
sleep 3

# Create model if needed
echo "Checking for tamil-llama model..."
if ! ollama list | grep -q "tamil-llama"; then
    echo "Creating tamil-llama model..."
    ollama create tamil-llama -f /models/tamil-llama/Modelfile || echo "WARNING: Model creation failed"
else
    echo "tamil-llama model already exists."
fi

echo "Ollama is ready!"

# Keep running
trap "kill $OLLAMA_PID; exit 0" SIGTERM SIGINT
wait $OLLAMA_PID