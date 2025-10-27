# 🚀 Deployment Guide

Deploy your RAG chatbot to Streamlit Cloud so anyone can use it with your API keys.

## Overview

**Deployed App**: Users access your app without needing API keys (you provide them)
**Local Clone**: Developers clone the repo and use their own API keys

The code automatically detects which scenario it's running in.

---

## Quick Deployment to Streamlit Cloud

### Prerequisites

- [ ] GitHub account
- [ ] [Streamlit Cloud account](https://share.streamlit.io) (free)
- [ ] OpenAI API key with credits
- [ ] Pinecone API key
- [ ] Code pushed to GitHub

### Step 1: Prepare for Deployment

**Important**: Streamlit Cloud uses `requirements.txt`, not `pyproject.toml`. If you have a `pyproject.toml` file, rename it:

```bash
mv pyproject.toml pyproject.toml.backup
```

Then push to GitHub:

```bash
cd /path/to/chat_with_wiki_2
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Set main file path: `src/main.py`
5. Choose a custom URL (optional)

### Step 3: Add API Keys as Secrets

**This is critical** - In the deployment settings, click "Advanced settings" and add your secrets:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
PINECONE_API_KEY = "your-actual-key-here"
```

**Important**: Use this exact format with quotes!

### Step 4: Deploy

1. Click "Deploy"
2. Wait 2-3 minutes
3. Your app is live at `https://your-app-name.streamlit.app`

### Step 5: Test

1. Visit your app URL
2. Add a test webpage
3. Ask a question
4. Verify it works ✅

---

## Security & Cost Management

### Set Spending Limits

Protect yourself from unexpected costs:

1. **OpenAI Dashboard**:
   - Go to [platform.openai.com/account/billing/limits](https://platform.openai.com/account/billing/limits)
   - Set soft limit: $10/month
   - Set hard limit: $50/month
   - Enable email alerts at $5, $10, $25

2. **Pinecone Dashboard**:
   - Monitor usage at [app.pinecone.io](https://app.pinecone.io)
   - Check index size and query volume

### Monitor Usage

- **OpenAI**: [platform.openai.com/usage](https://platform.openai.com/usage)
- **Pinecone**: [app.pinecone.io](https://app.pinecone.io)
- **Streamlit**: Check app analytics in dashboard

### Cost Estimates

| Usage Level | Sessions/Month | Estimated Cost |
|-------------|----------------|----------------|
| Light (demo) | 20 | $1-5 |
| Medium (shared) | 100 | $10-20 |
| Heavy (popular) | 500 | $50-100 |

Per session (5 questions about 1 webpage):
- Embeddings: ~$0.001
- GPT-4: ~$0.05-0.10
- Pinecone: ~$0.001
- **Total**: ~$0.05-0.11

### Reduce Costs

1. **Switch to GPT-3.5-Turbo** (~90% cheaper):
   ```python
   # In src/main.py
   llm = ChatOpenAI(
       model_name='gpt-3.5-turbo',  # instead of 'gpt-4'
       temperature=0.7,
       openai_api_key=get_api_key('OPENAI_API_KEY')
   )
   ```

2. **Add Usage Notice** in sidebar:
   ```python
   st.info("""
   ℹ️ Free demo using my API keys.
   For heavy usage, please clone the repo!
   """)
   ```

3. **Add Rate Limiting** (optional):
   ```python
   if 'last_query_time' not in st.session_state:
       st.session_state.last_query_time = 0

   if time.time() - st.session_state.last_query_time < 5:
       st.warning("Please wait 5 seconds between queries")
       st.stop()
   st.session_state.last_query_time = time.time()
   ```

---

## 🛠️ Troubleshooting

### Build Error: "Readme file does not exist" or "No pyproject.toml found"

These errors happen when Streamlit Cloud's UV tries to use `pyproject.toml`.

**Solution**:
```bash
# 1. Rename or delete pyproject.toml
mv pyproject.toml pyproject.toml.backup

# 2. Create empty packages.txt to force pip usage
touch packages.txt

# 3. Commit and push
git add .
git commit -m "Fix: Force pip installation on Streamlit Cloud"
git push
```

The `packages.txt` file tells Streamlit to use pip instead of UV.

### "Secrets not found" Error

1. Go to Streamlit Cloud dashboard
2. Click your app → Settings → Secrets
3. Verify TOML format with quotes
4. Click "Save" and reboot app

### "API key invalid" Error

- Check for typos or extra spaces in secrets
- Verify keys haven't expired
- Ensure OpenAI account has credits
- Try regenerating the keys

### App is Slow or Timing Out

- First run creates Pinecone index (~10 seconds - normal)
- Check if Pinecone index was created successfully
- Try smaller webpages for testing
- Check Streamlit Cloud resource limits

### High Unexpected Costs

1. Check OpenAI usage dashboard for unusual activity
2. Add spending limits immediately
3. Implement rate limiting
4. Consider switching to GPT-3.5

---

## 🔒 Security Best Practices

### Protect Your Keys

1. **Never** commit `.env` or `secrets.toml` to Git
2. **Always** use Streamlit secrets for deployed apps
3. **Set** hard spending limits on all APIs
4. **Monitor** usage dashboards regularly
5. **Rotate** keys if you suspect exposure

### If Keys Are Exposed

1. Immediately revoke keys in OpenAI/Pinecone dashboards
2. Generate new keys
3. Update Streamlit Cloud secrets
4. Review billing for unauthorized usage
5. Enable alerts for future issues

---

## 📋 Pre-Deployment Checklist

Before going public:

- [ ] Code is tested locally
- [ ] API keys work and have credits
- [ ] Spending limits are set
- [ ] App handles errors gracefully
- [ ] Usage instructions are clear in UI
- [ ] README includes live demo link
- [ ] Monitoring is set up
- [ ] Tested from different device/browser

---

## 🎉 Post-Deployment

### Update Documentation

Add to your README.md:

```markdown
## 🌐 Live Demo

Try it: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

*Note: Uses my personal API keys. For heavy usage, clone the repo.*
```

### Share on LinkedIn

Example post template:

```
🚀 Excited to share my RAG-powered chatbot!

Built with:
• OpenAI GPT-4
• Pinecone vector database
• LangChain for orchestration
• Streamlit for UI

Try it: [your-link]
Code: [your-repo]

#AI #MachineLearning #RAG #Python
```

### Monitor & Iterate

- Watch Streamlit analytics for usage patterns
- Check API costs weekly
- Respond to issues on GitHub
- Gather feedback from users
- Update based on learnings

---

## 📚 Additional Deployment Options

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/main.py", "--server.port=8501"]
```

Build and run:
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 --env-file .env rag-chatbot
```

### Heroku

Create `Procfile`:
```
web: streamlit run src/main.py --server.port=$PORT --server.address=0.0.0.0
```

Deploy:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
heroku config:set PINECONE_API_KEY=your-key
git push heroku main
```

---

## 🆘 Need Help?

- Check [API_KEY_SETUP.md](API_KEY_SETUP.md) for configuration details
- Review [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-cloud)
- Open an issue on GitHub
- Contact the maintainer

---

**Next**: Share your deployment and gather feedback! 🎉
