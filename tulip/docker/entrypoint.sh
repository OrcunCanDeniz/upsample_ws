#!/bin/bash

# Exit on any error
set -e

echo "Starting entrypoint script..."

# Check if BEVDepth directory exists
if [ -d "/workspace/BEVDepth" ]; then
    echo "Found BEVDepth directory, building..."
    cd /workspace/BEVDepth
    
    # Build BEVDepth in editable mode
    echo "Installing BEVDepth in editable mode..."
    pip install -e .
    
    echo "BEVDepth build completed successfully!"
else
    echo "Warning: BEVDepth directory not found at /workspace/BEVDepth"
fi

# Execute the command passed to docker run
echo "Executing: $@"
exec "$@"
