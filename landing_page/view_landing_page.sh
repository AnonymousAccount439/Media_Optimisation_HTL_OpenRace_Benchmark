#!/bin/bash

# Script to easily view the landing page with real data

echo "=========================================="
echo "Active Learning Landing Page"
echo "=========================================="
echo ""
echo "Starting local web server..."
echo "The landing page will be available at:"
echo ""
echo "  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start Python HTTP server
python3 -m http.server 8000

