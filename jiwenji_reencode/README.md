# Re-encode to H.265 (jiwenji_reencode)

GPU-accelerated H.265 (HEVC) re-encoding plugin for Stash AI Overhaul. Adds "Re-encode to H.265" actions to the Stash scenes page.

## Architecture

This plugin has two parts:

1. **Plugin** (runs in the AI Overhaul backend) — handles Stash integration, task management, and UI actions
2. **Worker sidecar** (separate Docker container) — runs ffmpeg with GPU access and write access to media files

The backend container stays read-only and GPU-free. Only the worker gets the dangerous bits.

## Prerequisites

- NVIDIA GPU with NVENC support (GTX 10-series or newer)
- Docker with NVIDIA Container Toolkit installed ([setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- Media files accessible from WSL2 / Docker

## Setup

### 1. Install the plugin

Copy the `jiwenji_reencode` folder into your AI Overhaul server's `plugins/` directory:

```
Stash-AIServer/
├── plugins/
│   ├── jiwenji_reencode/   ← this plugin
│   └── skier_aitagging/
```

### 2. Add the worker sidecar to docker-compose.yml

Add this service to your existing `Stash-AIServer/docker-compose.yml`:

```yaml
  reencode_worker:
    build:
      context: .
      dockerfile: Dockerfile.reencode
    image: stash-ai-server-reencode:local
    container_name: ai-overhaul-reencode-worker
    restart: always
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    env_file:
      - ./config.env
    environment:
      - REENCODE_WORKER=1
    volumes:
      - ./data:/app/data:rw
      - ./plugins:/app/plugins:rw
      - /mnt/z/f:/app/stash:rw        # ← YOUR media path, must be :rw
      - ./backend/stash_ai_server:/usr/local/lib/python3.14/site-packages/stash_ai_server:ro
```

> **Important:** Change `/mnt/z/f:/app/stash:rw` to match where your media lives in WSL2.

### 3. Create the worker Dockerfile

Create `Stash-AIServer/Dockerfile.reencode`:

```dockerfile
FROM stash-ai-server:local
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir aiohttp
COPY reencode_worker /app/reencode_worker
EXPOSE 4154
CMD ["python", "-m", "reencode_worker.server"]
```

### 4. Create the worker server

Copy the `reencode_worker/` directory into `Stash-AIServer/`:

```
Stash-AIServer/
├── reencode_worker/
│   ├── __init__.py
│   └── server.py
├── Dockerfile.reencode
└── docker-compose.yml
```

### 5. Build and start

```bash
cd /mnt/c/Users/YourUser/Documents/stash_tagging/Stash-AIServer

# Build the worker image
docker compose build reencode_worker

# Start the worker (won't affect other containers)
docker compose up -d reencode_worker

# Restart backend to pick up the new plugin
docker compose restart backend_prod
```

### 6. Configure path mappings

In the Stash AI Overhaul settings UI, configure **Path Mappings** for the `jiwenji_reencode` plugin:

| Source | Target | Slash Mode |
|--------|--------|------------|
| `Z:\f\` | `/app/stash/` | unix |

This translates Stash's Windows paths to where the worker container can find the files.

## Usage

Once set up, you'll see these actions on the Stash scenes page:

- **Re-encode to H.265** — single scene (detail view)
- **Re-encode Selected** — selected scenes
- **Re-encode Page** — all visible scenes
- **Re-encode All Scenes** — entire library

## Settings

All settings are configurable in the AI Overhaul plugin settings UI.

| Setting | Default | Description |
|---------|---------|-------------|
| Worker URL | `http://localhost:4154` | URL of the reencode_worker sidecar |
| Quality Level (CQ) | 28 | NVENC quality (0–51, lower = better) |
| Low-Bitrate CQ | 34 | CQ for already-compact files |
| NVENC Preset | p7 | p1 = fastest, p7 = best compression |
| Skip HEVC Files | true | Skip files already in H.265 |
| Delete Original | true | Delete original after successful encode |
| Output Suffix | *(empty)* | e.g. `_hevc`. Empty = replace in-place |
| Min Savings % | 15 | Reject encodes below this savings threshold |
| Max Concurrent | 0 | 0 = auto-detect from GPU |
| GPU Index | 0 | For multi-GPU systems |
| Enable Retries | true | Retry with more aggressive CQ on failure |
| Copy Metadata on Suffix | true | Copy tags/performers/etc. when using suffix mode |
| Tag After Re-encode | false | Chain into AI tagging after successful encode |

## Troubleshooting

### "worker unreachable" in connectivity check
- Is the reencode_worker container running? `docker ps | grep reencode`
- Check logs: `docker logs ai-overhaul-reencode-worker`

### "File not found" errors
- Check your **Path Mappings** setting — Stash sends Windows paths, the worker needs Linux paths
- Verify the media volume mount in docker-compose.yml is correct and `:rw`

### "ffmpeg_missing" in health check
- The worker Dockerfile didn't install ffmpeg properly. Rebuild: `docker compose build --no-cache reencode_worker`

### Encode starts but immediately fails
- Check GPU access: `docker exec ai-overhaul-reencode-worker nvidia-smi`
- If that fails, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Encode succeeds but Stash doesn't see the new file
- The rescan path must be the **Stash path** (Windows), not the container path
- Check that Stash's scan settings include the directory containing your media
