#!/bin/bash

# RAG Chatbot Setup Script
# This script sets up the development environment using UV (or falls back to pip)

set -e  # Exit on error

echo "🚀 RAG Chatbot Setup Script"
echo "============================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if UV is installed
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ UV is installed${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}⚠ UV is not installed${NC}"
    echo ""
    read -p "Would you like to install UV? (recommended, 10x faster) [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo -e "${BLUE}Installing UV...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if command -v uv &> /dev/null; then
            echo -e "${GREEN}✓ UV installed successfully${NC}"
            USE_UV=true
        else
            echo -e "${YELLOW}⚠ UV installation failed, falling back to pip${NC}"
            USE_UV=false
        fi
    else
        echo -e "${BLUE}Using pip instead${NC}"
        USE_UV=false
    fi
fi

echo ""

# Remove old virtual environments
if [ -d "venv" ]; then
    echo -e "${BLUE}Removing old venv...${NC}"
    rm -rf venv
fi

if [ -d ".venv" ]; then
    echo -e "${BLUE}Removing old .venv...${NC}"
    rm -rf .venv
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ "$USE_UV" = true ]; then
    uv venv
    VENV_PATH=".venv"
else
    python3 -m venv venv
    VENV_PATH="venv"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
if [ "$USE_UV" = true ]; then
    echo "  Using UV (fast!)..."
    uv pip install -r requirements.txt
else
    echo "  Using pip..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo ""
    read -p "Would you like to create .env from template? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file${NC}"
        echo ""
        echo -e "${YELLOW}⚠ IMPORTANT: Edit .env and add your API keys:${NC}"
        echo "  - OPENAI_API_KEY"
        echo "  - PINECONE_API_KEY"
        echo ""
        echo "Get your keys from:"
        echo "  - OpenAI: https://platform.openai.com/api-keys"
        echo "  - Pinecone: https://app.pinecone.io/"
        echo ""
    fi
else
    echo -e "${GREEN}✓ .env file found${NC}"
fi

# Display next steps
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Activate the virtual environment:"
if [ "$USE_UV" = true ]; then
    echo "   source .venv/bin/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Edit .env with your API keys (if you haven't already)"
echo ""
echo "3. Run the application:"
echo "   streamlit run src/main.py"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"
