import redis.asyncio as redis
import json

# Redis connection
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

STREAM_KEY = "detections_stream"


# ── PUSH detection to Redis ─────────────────────────────
async def push_detection(camera_id: str, detection: dict):
    await r.xadd(
        STREAM_KEY,
        {
            "camera_id": camera_id,
            "data": json.dumps(detection)
        },
        maxlen=10000
    )


# ── GET latest detections (for replay) ──────────────────
async def get_recent_detections(count: int = 50):
    items = await r.xrevrange(STREAM_KEY, count=count)
    return items