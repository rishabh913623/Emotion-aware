# 🚀 Backend Deployment Guide - Render

## Step-by-Step Deployment Instructions

### ✅ Prerequisites Completed
- [x] `requirements.txt` created with production dependencies
- [x] `Procfile` created for Render
- [x] `render.yaml` configuration file
- [x] `runtime.txt` specifying Python version
- [x] Backend configured for production (CORS, PORT, environment variables)
- [x] Code pushed to GitHub

---

## 📋 Step 1: Prepare Render Account

1. **Go to [Render.com](https://render.com)**
2. **Sign up** with your GitHub account
3. **Authorize Render** to access your repositories

---

## 📋 Step 2: Create New Web Service

1. **Click "New +" → "Web Service"**

2. **Connect Repository:**
   - Select: `rishabh913623/Emotion-aware`
   - Click "Connect"

3. **Configure Service:**

   | Setting | Value |
   |---------|-------|
   | **Name** | `emotion-aware-backend` |
   | **Region** | Oregon (US West) or closest to you |
   | **Branch** | `main` |
   | **Root Directory** | *(leave empty)* |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn -w 4 -k uvicorn.workers.UvicornWorker run_backend:app --bind 0.0.0.0:$PORT` |

4. **Select Plan:**
   - Choose **"Free"** (for testing)
   - Or **"Starter"** ($7/month) for better performance

---

## 📋 Step 3: Configure Environment Variables

In the Render dashboard, add these environment variables:

### Required Variables:

| Key | Value | Description |
|-----|-------|-------------|
| `PYTHON_VERSION` | `3.9.18` | Python version |
| `PORT` | `10000` | Port (auto-assigned by Render) |
| `ENVIRONMENT` | `production` | Environment name |
| `ALLOWED_ORIGINS` | `https://your-frontend.vercel.app,http://localhost:3001` | CORS origins (update after frontend deploy) |

### How to Add:
1. Scroll to "Environment" section
2. Click "Add Environment Variable"
3. Enter Key and Value
4. Click "Save Changes"

---

## 📋 Step 4: Deploy

1. **Click "Create Web Service"**
2. **Wait for deployment** (5-10 minutes for first deploy)
3. **Monitor logs** in the "Logs" tab
4. **Look for success message:**
   ```
   🚀 Starting Emotion-Aware Virtual Classroom Backend
   📊 Environment: production
   🔗 Server will be available at: http://0.0.0.0:10000
   ```

---

## 📋 Step 5: Test Your Backend

Once deployed, you'll get a URL like: `https://emotion-aware-backend.onrender.com`

### Test Endpoints:

```bash
# Health check
curl https://emotion-aware-backend.onrender.com/health

# API documentation
open https://emotion-aware-backend.onrender.com/docs

# Classroom page
open https://emotion-aware-backend.onrender.com/classroom
```

**Expected Response:**
```json
{
  "status": "healthy",
  "environment": "production",
  "active_rooms": 0,
  "total_users": 0,
  "features": {
    "real_users": true,
    "webrtc_streaming": true,
    "emotion_recognition": true,
    "real_time_dashboard": true
  }
}
```

---

## 📋 Step 6: Update CORS After Frontend Deployment

After deploying frontend to Vercel:

1. **Go to Render Dashboard** → Your Service
2. **Click "Environment"**
3. **Update `ALLOWED_ORIGINS`:**
   ```
   https://your-app.vercel.app,https://your-app-*.vercel.app,http://localhost:3001
   ```
4. **Click "Save Changes"** (triggers auto-redeploy)

---

## 🔧 Troubleshooting

### Issue: Build Fails

**Solution:**
- Check "Logs" tab for errors
- Ensure `requirements.txt` is in root directory
- Verify Python version compatibility

### Issue: WebSocket Connection Fails

**Solution:**
- Render Free tier has WebSocket limitations
- Upgrade to Starter plan ($7/mo)
- Or use alternative for WebSockets

### Issue: 502 Bad Gateway

**Solution:**
- Check if app is binding to `0.0.0.0:$PORT`
- Verify gunicorn command is correct
- Check logs for startup errors

### Issue: CORS Errors

**Solution:**
- Add your frontend URL to `ALLOWED_ORIGINS`
- Include both production and preview URLs
- Use wildcard for Vercel: `https://*.vercel.app`

---

## 📊 Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Web service created and configured
- [ ] Environment variables set
- [ ] First deployment completed successfully
- [ ] Health endpoint returns 200
- [ ] API docs accessible
- [ ] Classroom page loads
- [ ] CORS configured for frontend
- [ ] WebSocket connections tested

---

## 🎯 Your Backend URLs

After deployment, save these URLs:

- **Main API**: `https://emotion-aware-backend.onrender.com`
- **Health Check**: `https://emotion-aware-backend.onrender.com/health`
- **API Docs**: `https://emotion-aware-backend.onrender.com/docs`
- **Classroom**: `https://emotion-aware-backend.onrender.com/classroom`
- **WebSocket**: `wss://emotion-aware-backend.onrender.com/ws/classroom/{room_id}`

---

## 🔄 Continuous Deployment

Render automatically redeploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update feature"
git push origin main

# Render auto-deploys in 2-3 minutes
```

---

## 💰 Cost Estimate

| Plan | Cost | Features |
|------|------|----------|
| **Free** | $0/month | 750 hours/month, sleeps after 15min inactivity |
| **Starter** | $7/month | Always on, better performance, no sleep |
| **Standard** | $25/month | More resources, scaling |

**Recommendation:** Start with Free, upgrade to Starter if needed.

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Status**: https://status.render.com
- **Community**: https://community.render.com

---

**Last Updated:** February 8, 2026  
**Status:** ✅ Production Ready  
**Next Step:** Deploy Frontend to Vercel
