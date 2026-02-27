# Deployment Guide - AI Handwriting Recognition

This guide provides step-by-step instructions for deploying the handwriting recognition app on various platforms.

## 🎯 Quick Deployment Options

### 1️⃣ Hugging Face Spaces (FREE - Recommended)

**Best for:** Quick deployment with Gradio interface

**Steps:**
1. Create account at https://huggingface.co/join
2. Go to https://huggingface.co/new-space
3. Fill in:
   - Space name: `handwriting-recognition`
   - License: `MIT`
   - SDK: `Gradio`
   - Hardware: `CPU basic` (free)
4. Clone the space:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/handwriting-recognition
   cd handwriting-recognition
   ```
5. Copy these files to the space directory:
   - `app_gradio.py` → rename to `app.py`
   - `download_model.py`
   - `requirements.txt`
   - `utils/` folder
   - `README_HF.md` → rename to `README.md`

6. Create `.gitignore`:
   ```
   __pycache__/
   uploads/
   models/
   ```

7. Commit and push:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

8. Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/handwriting-recognition`

---

### 2️⃣ Streamlit Cloud (FREE)

**Best for:** Clean, modern interface

**Steps:**
1. Create account at https://streamlit.io/cloud
2. Create a GitHub repository and push these files:
   - `app_streamlit.py` → rename to `streamlit_app.py`
   - `requirements.txt`
   - `download_model.py`
   - `utils/`

3. Add `streamlit==1.28.0` to requirements.txt

4. In Streamlit Cloud:
   - Click "New app"
   - Select your repository
   - Main file: `streamlit_app.py`
   - Deploy!

5. App will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

### 3️⃣ Render (FREE Tier)

**Best for:** Flask production deployment

**Steps:**
1. Create account at https://render.com
2. Push code to GitHub
3. In Render dashboard:
   - Click "New" → "Web Service"
   - Connect your repository
   - Settings:
     - Name: `handwriting-ocr`
     - Environment: `Python 3`
     - Build Command: `pip install -r requirements.txt && python download_model.py`
     - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Click "Create Web Service"
5. Wait 5-10 minutes for deployment
6. App will be live at: `https://handwriting-ocr.onrender.com`

**Note:** Free tier sleeps after 15 min inactivity, takes ~30s to wake up.

---

### 4️⃣ Railway (FREE $5/month credit)

**Best for:** Fast deployment with database support

**Steps:**
1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python
5. Add environment variables (if needed)
6. Click "Deploy"
7. Generate domain in Settings → Networking

App will be live at: `https://YOUR_APP.up.railway.app`

---

### 5️⃣ Google Cloud Run (FREE tier)

**Best for:** Scalable containerized deployment

**Steps:**
1. Install Google Cloud CLI
2. Build and deploy:
   ```bash
   gcloud init
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/handwriting-ocr
   gcloud run deploy handwriting-ocr \
     --image gcr.io/YOUR_PROJECT_ID/handwriting-ocr \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi
   ```

---

### 6️⃣ Fly.io (FREE Tier)

**Steps:**
1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
2. Login: `flyctl auth login`
3. Launch app:
   ```bash
   flyctl launch
   ```
4. Answer prompts:
   - App name: handwriting-ocr
   - Region: Choose nearest
   - Database: No
5. Deploy: `flyctl deploy`

---

### 7️⃣ Local Deployment with Ngrok (FREE)

**Best for:** Testing and demos

**Steps:**
1. Install ngrok: https://ngrok.com/download
2. Start Flask app:
   ```bash
   python app.py
   ```
3. In another terminal:
   ```bash
   ngrok http 5000
   ```
4. Share the `https://xxxxx.ngrok.io` URL

**Note:** URL changes on restart, upgrade for persistent URLs.

---

### 8️⃣ Docker Deployment (Any Platform)

**Build and run locally:**
```bash
docker build -t handwriting-ocr .
docker run -p 5000:5000 handwriting-ocr
```

**For Gradio version:**
```bash
docker build -f Dockerfile.gradio -t handwriting-ocr-gradio .
docker run -p 7860:7860 handwriting-ocr-gradio
```

---

## 📊 Deployment Comparison

| Platform | Cost | Setup Time | Best For |
|----------|------|------------|----------|
| **Hugging Face Spaces** | FREE | 5 min | Quick sharing, Gradio UI |
| **Streamlit Cloud** | FREE | 5 min | Clean modern UI |
| **Render** | FREE* | 10 min | Production Flask apps |
| **Railway** | $5 credit | 5 min | Fast deployment |
| **Google Cloud Run** | FREE* | 15 min | Scalable containers |
| **Fly.io** | FREE* | 10 min | Global edge deployment |
| **Ngrok** | FREE | 2 min | Local testing/demos |

*Free tier available with limitations

---

## 🔧 Troubleshooting

### Model Download Issues
If deployment fails during model download:

**Option 1:** Pre-download model locally and commit to repo (not recommended for Git, files are large)

**Option 2:** Use Hugging Face model hub directly:
```python
# In app.py, change MODEL_PATH to:
MODEL_PATH = "microsoft/trocr-base-handwritten"
```

### Memory Issues
If deployment runs out of memory:
- Use smaller model: `microsoft/trocr-small-handwritten`
- Increase memory allocation in platform settings
- Use CPU-only torch for smaller footprint

### Slow Cold Starts
Most free tiers "sleep" after inactivity:
- **Render**: ~30s wake time
- **Railway**: Instant wake
- **Hugging Face**: ~20s wake time

Consider upgrading to paid tier for always-on instances.

---

## 🚀 Recommended Deployment

**For quick demo:** Hugging Face Spaces (Gradio)
**For production:** Railway or Render
**For local testing:** Ngrok

---

## 📝 Post-Deployment Checklist

- ✅ Test image upload
- ✅ Verify OCR prediction works
- ✅ Test CER/WER calculation
- ✅ Check mobile responsiveness
- ✅ Monitor resource usage
- ✅ Set up error logging (optional)

---

Need help? Check the main README.md or open an issue!
