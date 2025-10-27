# 🚀 Getting Started

Get your RAG chatbot running in under 5 minutes. This guide covers everything from installation to your first chat.

## Prerequisites

- Python 3.10+ installed
- Git installed
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://www.pinecone.io/)

---

## Quick Setup (Recommended)

### 1️⃣ Install UV (Fast Package Manager)

UV is 10-100x faster than pip. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2️⃣ Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/rag-webpage-chatbot.git
cd rag-webpage-chatbot

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate     # macOS/Linux
# or
.venv\Scripts\activate        # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 3️⃣ Configure API Keys

```bash
# Copy template
cp .env.example .env

# Edit with your keys
nano .env
```

Add your API keys:
```env
OPENAI_API_KEY=sk-proj-your-key-here
PINECONE_API_KEY=your-key-here
```

### 4️⃣ Run the App

```bash
streamlit run src/main.py
```

The app opens automatically at `http://localhost:8501`

---

## Alternative: Setup with Pip

If you prefer not to install UV:

```bash
# Clone repository
git clone https://github.com/yourusername/rag-webpage-chatbot.git
cd rag-webpage-chatbot

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate      # macOS/Linux
# or
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
nano .env

# Run
streamlit run src/main.py
```

---

## 🎯 First Use

1. **Add a webpage**: Enter a URL in the sidebar (try Wikipedia articles or documentation)
2. **Wait ~10-30 seconds**: While the page is processed and indexed
3. **Ask questions**: Chat about the content naturally
4. **Add more pages**: Load up to 5 webpages total
5. **Restart session**: Click the restart button to start fresh

### Example URLs to Try

- **Wikipedia**: `https://en.wikipedia.org/wiki/Machine_learning`
- **Documentation**: `https://docs.python.org/3/tutorial/`
- **Articles**: Any blog post or article URL

---

## Quick Command Reference

### Common UV Commands

```bash
# Create environment
uv venv

# Activate (do this every time)
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Update packages
uv pip install -r requirements.txt --upgrade

# Fresh start (if issues)
rm -rf .venv && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Run app
streamlit run src/main.py
```

### Common Pip Commands

```bash
# Create environment
python -m venv venv

# Activate
source venv/bin/activate

# Install
pip install -r requirements.txt

# Update
pip install -r requirements.txt --upgrade

# Run app
streamlit run src/main.py
```

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError"
```bash
# Ensure environment is activated
source .venv/bin/activate    # UV
# or
source venv/bin/activate     # pip

# Reinstall dependencies
uv pip install -r requirements.txt
```

### "API Key Error"
- Check `.env` file exists in project root
- Verify keys are correct (no extra spaces)
- Ensure keys have credits available

### "Port 8501 already in use"
```bash
# Kill other Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run src/main.py --server.port 8502
```

### "Pinecone Connection Error"
- Wait 10 seconds for index creation (first run only)
- Check internet connection
- Verify Pinecone API key is valid

### "Index creation taking too long"
The app creates a Pinecone index on first run. This is normal and takes ~10 seconds. Subsequent runs are instant.

---

## 📊 How It Works

### The RAG Pipeline

1. **Load** → Webpage content is fetched
2. **Chunk** → Text split into 1000-character pieces
3. **Embed** → Chunks converted to 384-dimensional vectors
4. **Store** → Vectors indexed in Pinecone (with session namespace)
5. **Query** → Your question is embedded
6. **Retrieve** → Top 3 most similar chunks found
7. **Generate** → GPT-4 creates answer from context

### Session Management

- Each session gets a unique namespace in Pinecone
- Load up to 5 webpages per session
- Restart button clears session and creates new namespace
- Multiple users don't interfere with each other

### Cost Estimates

Per session (5 questions):
- Embeddings: ~$0.001
- GPT-4 queries: ~$0.05-0.10
- Pinecone storage: ~$0.001
- **Total**: ~$0.05-0.11

---

## 💡 Pro Tips

1. **Specific URLs work best**: Use article/documentation pages, not homepages
2. **Long content is fine**: The model excels at summarization
3. **Ask follow-ups**: Test the conversation memory
4. **Check query processing**: Expand the "Query Processing Details" section to see how your question was refined
5. **Load related pages**: Add multiple pages on the same topic for better context

---

**Ready to deploy?** → [DEPLOYMENT.md](DEPLOYMENT.md)
**API key questions?** → [API_KEY_SETUP.md](API_KEY_SETUP.md)
