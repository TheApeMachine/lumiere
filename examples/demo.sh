#!/bin/bash

# Lumiere Demo Script
# This script demonstrates the complete Lumiere AI Music Video Generator pipeline

set -e

echo "=========================================="
echo "  Lumiere AI Music Video Generator Demo"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
echo -e "${BLUE}Checking server status...${NC}"
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "Error: Server is not running. Please start the server with: ./lumiere"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Create a test audio file
echo -e "${BLUE}Creating test audio file...${NC}"
echo "Dummy MP3 content for demonstration" > /tmp/demo_audio.mp3
echo -e "${GREEN}✓ Test audio created${NC}"
echo ""

# Optional: Create character images
echo -e "${BLUE}Creating test character images...${NC}"
echo "Character Image 1" > /tmp/character1.png
echo "Character Image 2" > /tmp/character2.png
echo -e "${GREEN}✓ Character images created${NC}"
echo ""

# Step 1: Create a project
echo -e "${YELLOW}Step 1: Creating project...${NC}"
PROJECT_JSON=$(curl -s -X POST http://localhost:8080/api/v1/projects \
  -F "audio=@/tmp/demo_audio.mp3" \
  -F "prompt=Cinematic journey through a mystical forest at sunset, with ethereal light beams" \
  -F "character_images=@/tmp/character1.png" \
  -F "character_images=@/tmp/character2.png")

PROJECT_ID=$(echo $PROJECT_JSON | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Failed to create project"
    echo $PROJECT_JSON
    exit 1
fi

echo -e "${GREEN}✓ Project created with ID: $PROJECT_ID${NC}"
echo ""

# Step 2: View project details
echo -e "${YELLOW}Step 2: Viewing project details...${NC}"
curl -s http://localhost:8080/api/v1/projects/$PROJECT_ID | python3 -m json.tool || echo $PROJECT_JSON
echo ""

# Step 3: Start processing
echo -e "${YELLOW}Step 3: Starting pipeline processing...${NC}"
PROCESS_RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/projects/$PROJECT_ID/process)
echo $PROCESS_RESPONSE | python3 -m json.tool || echo $PROCESS_RESPONSE
echo -e "${GREEN}✓ Processing started${NC}"
echo ""

# Step 4: Monitor progress
echo -e "${YELLOW}Step 4: Monitoring progress...${NC}"
for i in {1..10}; do
    echo -e "${BLUE}Checking status (attempt $i)...${NC}"
    STATUS=$(curl -s http://localhost:8080/api/v1/projects/$PROJECT_ID | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    echo "Current status: $STATUS"
    
    if [ "$STATUS" = "completed" ]; then
        echo -e "${GREEN}✓ Processing completed!${NC}"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "Error: Processing failed"
        exit 1
    fi
    
    sleep 1
done
echo ""

# Step 5: View final results
echo -e "${YELLOW}Step 5: Viewing final results...${NC}"
FINAL_JSON=$(curl -s http://localhost:8080/api/v1/projects/$PROJECT_ID)
echo $FINAL_JSON | python3 -m json.tool || echo $FINAL_JSON
echo ""

# Extract and display key information
echo -e "${BLUE}=== Project Summary ===${NC}"
echo "Project ID: $PROJECT_ID"
echo "Status: $(echo $FINAL_JSON | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"

# Count generated assets
KEY_MOMENTS=$(echo $FINAL_JSON | grep -o '"key_moments":\[' | wc -l)
if [ "$KEY_MOMENTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Concept generated with key moments${NC}"
fi

SEEDS=$(echo $FINAL_JSON | grep -o '"visual_seeds":\[' | wc -l)
if [ "$SEEDS" -gt 0 ]; then
    echo -e "${GREEN}✓ Visual seeds generated${NC}"
fi

ANIMATIONS=$(echo $FINAL_JSON | grep -o '"animations":\[' | wc -l)
if [ "$ANIMATIONS" -gt 0 ]; then
    echo -e "${GREEN}✓ Animations generated${NC}"
fi

FINAL_VIDEO=$(echo $FINAL_JSON | grep -o '"final_video":"[^"]*"' | cut -d'"' -f4)
if [ ! -z "$FINAL_VIDEO" ]; then
    echo -e "${GREEN}✓ Final video composed: $FINAL_VIDEO${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "  Demo completed successfully!"
echo "==========================================${NC}"
echo ""
echo "To view all projects:"
echo "  curl http://localhost:8080/api/v1/projects"
echo ""
echo "To view this project:"
echo "  curl http://localhost:8080/api/v1/projects/$PROJECT_ID"
