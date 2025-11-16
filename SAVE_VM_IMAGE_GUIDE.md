# How to Save a GCP VM Image

## Prerequisites
- Billing must be enabled on your GCP project
- VM must be stopped or in a stable state

## Steps to Save VM Image

### Option 1: Using gcloud CLI (Recommended)

```bash
# 1. Stop the VM
gcloud compute instances stop diaremot2 --zone=us-east1-c

# 2. Create image from VM
gcloud compute images create diaremot2-snapshot-20251109 \
  --source-disk=diaremot2 \
  --source-disk-zone=us-east1-c \
  --family=diaremot2 \
  --description="DiaRemot2 with Silero VAD optimizations"

# 3. Verify image created
gcloud compute images list --filter="name:diaremot2-snapshot*"

# 4. Restart VM
gcloud compute instances start diaremot2 --zone=us-east1-c
```

### Option 2: Using GCP Console

1. Go to **Google Cloud Console**
2. Navigate to **Compute Engine** → **Images**
3. Click **Create Image**
4. Fill in:
   - **Name**: `diaremot2-snapshot-20251109`
   - **Source**: Choose disk `diaremot2`
   - **Zone**: `us-east1-c`
   - **Family**: `diaremot2` (optional, for grouping)
5. Click **Create**

---

## What Gets Saved

✅ **Included in image:**
- All installed packages (.venv, models, libraries)
- Repository code (diaremot2-on/)
- Model files (Silero, ECAPA, etc.)
- Configuration files
- Cache and downloaded models

✅ **Storage cost:** ~$0.05/month per 10GB image

---

## Restoring from Image

```bash
# Create new VM from saved image
gcloud compute instances create diaremot2-restored \
  --image=diaremot2-snapshot-20251109 \
  --image-project=diaremot \
  --machine-type=n1-standard-8 \
  --zone=us-east1-c \
  --boot-disk-size=100GB
```

---

## Current Status

⚠️ **Billing Issue**
- Your project has billing disabled
- Need to enable billing before creating/managing VMs
- Enable at: https://console.cloud.google.com/billing/enable?project=diaremot

Once billing is enabled:
1. Run the gcloud commands above to create image
2. Image will be saved in your GCP project
3. You can delete the VM and restore later from image

---

## Estimate Costs

| Item | Monthly Cost |
|------|--------------|
| VM (n1-standard-8, stopped) | $0 |
| VM disk storage (100GB) | ~$5 |
| Image storage (50GB) | ~$2.50 |
| **Total (with image, VM off)** | **~$2.50/month** |

---

## Quick Commands Once Billing Enabled

```bash
# Stop VM
gcloud compute instances stop diaremot2 --zone=us-east1-c

# Save image
gcloud compute images create diaremot2-$(date +%Y%m%d) \
  --source-disk=diaremot2 \
  --source-disk-zone=us-east1-c

# Start VM again
gcloud compute instances start diaremot2 --zone=us-east1-c

# List your images
gcloud compute images list
```

---

## Recommended Workflow

1. ✅ Enable billing on GCP project
2. ✅ Stop the VM (`gcloud compute instances stop diaremot2 --zone=us-east1-c`)
3. ✅ Create image snapshot
4. ✅ Delete the VM to save costs
5. ✅ Keep image as backup
6. ✅ Restore VM from image whenever needed

This way you pay only for image storage (~$2.50/month) instead of VM costs (~$30/month).
