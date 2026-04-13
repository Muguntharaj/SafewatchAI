/**
 * CCTVFeedV1 — canvas-based MJPEG renderer (legacy / alternative to CCTVFeed).
 * The original file had a second useEffect nested INSIDE the first one's body,
 * which is invalid React and caused a runtime crash. Rewritten cleanly.
 *
 * NOTE: The primary CCTVFeed.tsx (img-tag version) is simpler and preferred.
 * This file is kept for reference / canvas-overlay use cases.
 */
import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Camera, Circle } from "lucide-react";
import { Badge } from "./ui/badge";

interface Detection {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  timestamp: Date;
}

interface CCTVFeedProps {
  cameraId: string;
  onDetection?: (detection: Detection) => void;
  isActive: boolean;
  compact?: boolean;
}

export function CCTVFeed({
  cameraId,
  onDetection,
  isActive,
  compact = false,
}: CCTVFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  // Draw MJPEG frames onto the canvas via polling
  useEffect(() => {
    if (!isActive) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let alive = true;
    const img = new Image();

    const drawFrame = () => {
      if (!alive) return;
      // Append timestamp to bust browser cache for each frame
      img.src = `http://localhost:8000/api/stream/${cameraId}?t=${Date.now()}`;
    };

    img.onload = () => {
      if (!alive) return;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    const intervalId = setInterval(drawFrame, 33); // ~30 FPS

    return () => {
      alive = false;
      clearInterval(intervalId);
    };
  }, [cameraId, isActive]);

  // WebSocket for real-time detection events
  useEffect(() => {
    if (!isActive || !onDetection) return;

    const ws = new WebSocket("ws://localhost:8000/ws/detections");
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.camera_id !== cameraId) return;

        const detections: any[] = data.detections ?? [];
        detections.forEach((d) => {
          onDetection({
            id: `${cameraId}-${Date.now()}-${Math.random()}`,
            x: d.bbox?.x1 ?? 0,
            y: d.bbox?.y1 ?? 0,
            width: (d.bbox?.x2 ?? 0) - (d.bbox?.x1 ?? 0),
            height: (d.bbox?.y2 ?? 0) - (d.bbox?.y1 ?? 0),
            confidence: d.confidence ?? 0,
            timestamp: new Date(),
          });
        });
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = () => ws.close();

    return () => {
      ws.close();
    };
  }, [cameraId, isActive, onDetection]);

  const canvasWidth = compact ? 640 : 1280;
  const canvasHeight = compact ? 360 : 720;

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden">
      <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
        <motion.div
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          <Circle className="w-3 h-3 fill-red-600 text-red-600" />
        </motion.div>
        <Badge variant="destructive" className="gap-1">
          <Camera className="w-3 h-3" />
          LIVE
        </Badge>
      </div>

      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        className="w-full h-full object-contain"
      />
    </div>
  );
}
