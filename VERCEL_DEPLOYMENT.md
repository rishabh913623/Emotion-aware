# 🚀 Frontend Deployment Guide - Vercel

## Prerequisites
- ✅ Backend deployed on Render
- ✅ Backend URL ready (e.g., https://emotion-aware-backend-xxxx.onrender.com)
- ✅ GitHub repository updated

---

## Step 1: Update Backend URL

**IMPORTANT:** Replace the backend URL in the config file with YOUR Render URL!

Edit `frontend/src/config/api.ts`:
```typescript
export const API_BASE_URL = 'https://YOUR-RENDER-URL.onrender.com'
export const WS_BASE_URL = 'wss://YOUR-RENDER-URL.onrender.com'
```

---

## Step 2: Test Frontend Locally (Optional)

```bash
cd frontend
npm install
npm run build  # Test if build works
npm run dev    # Test locally
```

---

## Step 3: Deploy to Vercel

### Option A: Using Vercel Dashboard (Easiest)

1. **Go to:** https://vercel.com/login
2. **Sign in** with GitHub
3. **Click:** "Add New..." → "Project"
4. **Import** your repository: `rishabh913623/Emotion-aware`
5. **Configure:**

   ```
   Framework Preset: Vite
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

6. **Add Environment Variables:**
   - Click "Environment Variables"
   - Add these:
     ```
     VITE_API_URL = https://your-backend.onrender.com
     VITE_WS_URL = wss://your-backend.onrender.com
     ```

7. **Click:** "Deploy"
8. **Wait:** 2-3 minutes
9. **Done!** Your app is live at: `https://your-app.vercel.app`

---

### Option B: Using Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend
cd frontend

# Login to Vercel
vercel login

# Deploy
vercel --prod

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No  
# - Project name? emotion-aware-classroom
# - In which directory? ./
# - Override settings? No
```

---

## Step 4: Update Backend CORS

After frontend deployment, update your Render backend:

1. **Go to:** Render Dashboard → Your Backend Service
2. **Click:** "Environment"
3. **Update `ALLOWED_ORIGINS`:**
   ```
   https://your-app.vercel.app,https://your-app-*.vercel.app,http://localhost:3001
   ```
4. **Click:** "Save Changes" (auto-redeploys)

---

## Step 5: Test Your Deployed App

Visit your Vercel URL and test:

- ✅ Login page loads
- ✅ Can create account
- ✅ Can join classroom
- ✅ Chat works
- ✅ Video/audio controls work
- ✅ Attendance tracking works
- ✅ Screen sharing works

---

## Configuration Files Created

- ✅ `frontend/src/config/api.ts` - API configuration
- ✅ `frontend/vercel.json` - Vercel settings
- ✅ `frontend/.vercelignore` - Files to ignore
- ✅ `frontend/.env.example` - Environment template

---

## Environment Variables

Set these in Vercel Dashboard under "Settings" → "Environment Variables":

| Variable | Value | Environment |
|----------|-------|-------------|
| `VITE_API_URL` | `https://your-backend.onrender.com` | Production |
| `VITE_WS_URL` | `wss://your-backend.onrender.com` | Production |

---

## Troubleshooting

### Build Fails
- Check Node version (requires 16+)
- Verify all dependencies in package.json
- Check build logs in Vercel

### API Connection Fails
- Verify backend URL is correct
- Check CORS settings on backend
- Ensure backend is running (check Render)

### WebSocket Not Working
- Verify WSS URL (not WS)
- Check Render backend logs
- Ensure backend has WebSocket support

---

## Custom Domain (Optional)

1. **Vercel Dashboard** → Project → "Settings" → "Domains"
2. **Add domain:** `classroom.yourdomain.com`
3. **Update DNS** as instructed by Vercel
4. **Update Render CORS** with new domain

---

## Continuous Deployment

Vercel auto-deploys on every push to GitHub:

```bash
git add .
git commit -m "Update frontend"
git push origin main
# Vercel auto-deploys in 1-2 minutes
```

---

## Quick Deploy Commands

```bash
# One-time setup
cd frontend
npm i -g vercel
vercel login

# Deploy
vercel --prod

# View logs
vercel logs

# Open app
vercel --prod --open
```

---

## Your URLs After Deployment

- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-backend.onrender.com`
- **API Docs**: `https://your-backend.onrender.com/docs`

---

**Status:** ✅ Ready to Deploy  
**Next:** Go to Vercel and deploy now!
