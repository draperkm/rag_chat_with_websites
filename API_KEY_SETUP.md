# 🔑 API Key Setup

Understanding how API keys work in this application for both deployment and local development.

## Two Scenarios

### 1. Deployed App (Streamlit Cloud)
**You deploy → Users access without keys**

- Visitors don't need API keys
- You provide keys via Streamlit secrets
- You pay for all usage
- Best for demos and portfolios

### 2. Local Development
**Developers clone → Use their own keys**

- Each developer provides their own keys
- Keys stored in local `.env` file
- Each person pays for their own usage
- Keys never leave their machine

---

## How It Works

The app uses intelligent fallback logic in [`src/utils.py`](src/utils.py):

```python
def get_api_key(key_name):
    """Get key from Streamlit secrets (deployed) or .env (local)."""
    try:
        # Try Streamlit secrets first (for deployed app)
        return st.secrets[key_name]
    except (AttributeError, KeyError, FileNotFoundError):
        # Fall back to environment variables (for local)
        return os.environ.get(key_name)
```

**Result**: Same code works in both scenarios - no changes needed!

---

## Setup for Streamlit Cloud (Deployment)

### Get Your API Keys

1. **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **Pinecone**: [app.pinecone.io](https://app.pinecone.io/)

### Add to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Open your app → Settings → Advanced settings
3. In "Secrets" section, add:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key"
PINECONE_API_KEY = "your-actual-key"
```

4. Save and reboot app

### Verify

Visit your app and test with a webpage. If it works without asking for keys → Success!

---

## Setup for Local Development

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
# With UV (recommended)
uv venv
source .venv/bin/activate

# Or with pip
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
# With UV
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
# Copy template
cp .env.example .env

# Edit with your editor
nano .env
# or: code .env, vim .env, etc.
```

Add your keys:
```env
OPENAI_API_KEY=sk-your-personal-key-here
PINECONE_API_KEY=your-personal-key-here
```

### 5. Run

```bash
streamlit run src/main.py
```

App automatically uses keys from your `.env` file.

---

## 🔒 Security

### What's Protected

| File | In Git? | Purpose |
|------|---------|---------|
| `.env` | ❌ No | Your real keys (local) |
| `.env.example` | ✅ Yes | Template only |
| `.streamlit/secrets.toml` | ❌ No | Your real keys (deployed) |
| `.streamlit/secrets.toml.example` | ✅ Yes | Template only |
| `src/main.py` | ✅ Yes | Uses `get_api_key()` function |
| `src/utils.py` | ✅ Yes | Has `get_api_key()` function |

### Security Checklist

- [ ] `.env` is in `.gitignore`
- [ ] Never commit real API keys
- [ ] Spending limits set on APIs
- [ ] Monitoring enabled

---

## 💰 Cost Management

### For Deployed Apps (Your Keys)

**You pay for all usage**. Protect yourself:

#### Set OpenAI Limits

Visit [platform.openai.com/account/billing/limits](https://platform.openai.com/account/billing/limits):
- Soft limit: $10/month
- Hard limit: $50/month
- Email alerts: At $5, $10, $25

#### Monitor Usage

- **OpenAI**: [platform.openai.com/usage](https://platform.openai.com/usage)
- **Pinecone**: [app.pinecone.io](https://app.pinecone.io)
- Check weekly

#### Add Rate Limiting

Optional but recommended:

```python
# In src/main.py
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if st.session_state.query_count > 100:
    st.error("Demo limit reached. Please clone the repo!")
    st.stop()
```

### Estimated Costs

| Scenario | Sessions/Month | Cost/Month |
|----------|----------------|------------|
| Personal demo | 20 | $1-5 |
| Shared demo | 100 | $10-20 |
| Popular app | 500 | $50-100 |

**Per session** (5 questions, 1 webpage): ~$0.05-0.11

### Ways to Reduce Costs

1. **Use GPT-3.5 instead of GPT-4** (90% cheaper)
2. **Add usage instructions** in UI
3. **Implement rate limiting**
4. **Set hard spending limits**

### For Local Development (Their Keys)

Each developer pays for their own usage. No cost to you!

---

## 🧪 Testing Your Setup

### Test Streamlit Secrets

After deploying, check logs:
```python
# Temporarily add to main.py
import streamlit as st
st.write("OpenAI key loaded:", bool(st.secrets.get("OPENAI_API_KEY")))
st.write("Pinecone key loaded:", bool(st.secrets.get("PINECONE_API_KEY")))
```

### Test Local .env

```bash
# Verify keys load
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Pinecone:', bool(os.getenv('PINECONE_API_KEY')))"
```

Should print `True` for both.

---

## ❓ FAQ

### Q: Can I use different keys for local vs deployed?
**A**: Yes! That's the design. Streamlit Cloud uses secrets, local uses .env.

### Q: What if I want users to provide their own keys?
**A**: Remove Streamlit secrets and modify UI:

```python
with st.sidebar:
    user_key = st.text_input("Enter OpenAI API key", type="password")
    if not user_key:
        st.warning("Please provide an API key")
        st.stop()
```

### Q: How do I rotate my API keys?
**A**:
1. Generate new keys in OpenAI/Pinecone dashboards
2. Update Streamlit Cloud secrets
3. Reboot app
4. Delete old keys

### Q: What if my keys are exposed?
**A**:
1. **Immediately** revoke in OpenAI/Pinecone dashboards
2. Generate new keys
3. Update Streamlit secrets
4. Review billing for unauthorized usage

### Q: Can I see who's using my deployed app?
**A**: Yes! Streamlit Cloud provides analytics (visitor count, usage patterns).

---

## 📚 Additional Resources

- [Streamlit Secrets Docs](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [OpenAI API Best Practices](https://platform.openai.com/docs/guides/production-best-practices)
- [Pinecone Security](https://docs.pinecone.io/docs/security)

---

## ✅ Quick Checklist

### For Deployment:
- [ ] OpenAI key added to Streamlit secrets
- [ ] Pinecone key added to Streamlit secrets
- [ ] Spending limits set
- [ ] Tested with sample URL
- [ ] Monitoring enabled

### For Local Development:
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `.env` created from `.env.example`
- [ ] API keys added to `.env`
- [ ] App runs successfully

---

**Need help?** See [GETTING_STARTED.md](GETTING_STARTED.md) or [DEPLOYMENT.md](DEPLOYMENT.md)
