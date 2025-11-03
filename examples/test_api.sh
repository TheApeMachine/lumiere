#!/bin/bash

# Test script for Lumiere API
# This script demonstrates how to use the Lumiere API

BASE_URL="http://localhost:8080"

echo "=== Lumiere API Test Script ==="
echo ""

# Test 1: Health Check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq .
echo ""
echo ""

# Test 2: Create Project
echo "2. Creating a test project..."
# Note: Replace with actual audio file path
PROJECT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/projects" \
  -F "prompt=Epic cinematic landscape with mountains and sunset" \
  -F "audio=@test_audio.mp3" 2>/dev/null || echo '{"error": "Please create a test_audio.mp3 file"}')

echo "$PROJECT_RESPONSE" | jq .
PROJECT_ID=$(echo "$PROJECT_RESPONSE" | jq -r '.id')
echo ""
echo ""

if [ "$PROJECT_ID" != "null" ] && [ "$PROJECT_ID" != "" ]; then
    # Test 3: Get Project Status
    echo "3. Getting project status..."
    curl -s "$BASE_URL/api/v1/projects/$PROJECT_ID" | jq .
    echo ""
    echo ""

    # Test 4: Start Processing
    echo "4. Starting pipeline processing..."
    curl -s -X POST "$BASE_URL/api/v1/projects/$PROJECT_ID/process" | jq .
    echo ""
    echo ""

    # Test 5: Check status after processing starts
    echo "5. Checking status after processing starts..."
    sleep 2
    curl -s "$BASE_URL/api/v1/projects/$PROJECT_ID" | jq .
    echo ""
    echo ""
fi

# Test 6: List all projects
echo "6. Listing all projects..."
curl -s "$BASE_URL/api/v1/projects" | jq .
echo ""

echo "=== Test Complete ==="
