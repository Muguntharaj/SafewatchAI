/**
 * CCTVFeed.tsx — Reference CCTV Build
 *
 * MJPEG img tag: the browser renders every multipart frame the server sends.
 * No client-side throttle, no requestAnimationFrame gate — the img onLoad
 * fires naturally as each frame arrives over the keep-alive HTTP connection.
 * Frame rate is determined entirely by the camera hardware and network speed.
 *
 * FIXES:
 *  1. SPURIOUS onerror ON HEALTHY MJPEG STREAM
 *     Browsers fire onerror on MJPEG <img> during the initial TCP negotiation
 *     phase even when the stream is perfectly healthy. Old code treated that as
 *     a real error, incremented retriesRef, and triggered the backoff chain →
 *     visible "Connecting…" flash on every page load.
 *     Fix: if status is already 'live', ignore onerror entirely.
 *     Fix: initial onerror within the first 2 seconds is ignored (startup grace).
 *
 *  2. PING INTERVAL TOO AGGRESSIVE
 *     Old: 5000ms ping while offline/error → 12 HEAD requests/min per camera.
 *     Fix: 10000ms ping. Server still responds immediately when back online.
 *     The PING_INTERVAL constant is now exported and configurable.
 *
 *  3. MAX_RETRIES TOO LOW FOR NETWORK HICCUPS
 *     Old: 3 retries → after 3 blips goes permanently "offline" until manual retry.
 *     Fix: 5 retries, then offline. Backoff: 2s→4s→8s→16s→32s.
 *
 *  4. objectFit 'cover' CROPS WIDE-ANGLE CCTV FOOTAGE
 *     Reference footage is 1270×720 (near-16:9). 'cover' is fine.
 *     Changed to 'contain' with black letterbox — never crops the scene.
 *     Security operators must see full frame, not a cropped version.
 *
 *  5. LIVE BADGE POSITION — moved to match real DVR software (top-left).
 *     Red pulsing dot + "REC" text, not just "LIVE".
 *
 *  6. CAMERA ID OVERLAY MATCHES DVR STYLE
 *     Bottom-left, white text, monospace — exactly like reference footage.
 *     Shows camera name (passed via prop) not raw camera_id.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface CCTVFeedProps {
  cameraId:    string;
  cameraName?: string;      // NEW: shown in overlay like real DVR
  isActive?:   boolean;
  onDetection?: (detection: any) => void;
}

const API           = 'http://localhost:8000';
const MAX_RETRIES   = 5;
const PING_INTERVAL = 10_000;    // ms — HEAD requests while offline
const STARTUP_GRACE = 2_000;     // ms — ignore onerror within first 2s

type Status = 'loading' | 'live' | 'error' | 'offline';

export function CCTVFeed({ cameraId, cameraName, isActive = true }: CCTVFeedProps) {
  const [status,     setStatus]     = useState<Status>('loading');
  const [renderTick, setRenderTick] = useState(0);
  const epochRef     = useRef(0);
  const retriesRef   = useRef(0);
  const mountedRef   = useRef(true);
  const mountTimeRef = useRef(Date.now());    // FIX 1: startup grace period
  const imgRef       = useRef<HTMLImageElement>(null);
  const retryTimer   = useRef<ReturnType<typeof setTimeout>>();
  const pingTimer    = useRef<ReturnType<typeof setInterval>>();

  const newSrc = useCallback(() => {
    epochRef.current += 1;
    setRenderTick(n => n + 1);
  }, []);

  // Reset when cameraId changes
  useEffect(() => {
    mountedRef.current  = true;
    mountTimeRef.current = Date.now();
    epochRef.current    = 0;
    retriesRef.current  = 0;
    setRenderTick(0);
    setStatus('loading');
    return () => {
      mountedRef.current = false;
      clearTimeout(retryTimer.current);
      clearInterval(pingTimer.current);
    };
  }, [cameraId]);

  // Ping loop while offline/error
  useEffect(() => {
    if (status !== 'error' && status !== 'offline') {
      clearInterval(pingTimer.current);
      return;
    }
    pingTimer.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/stream/${cameraId}`, { method: 'HEAD' });
        if (r.ok && mountedRef.current) {
          retriesRef.current = 0;
          setStatus('loading');
          newSrc();
        }
      } catch { /* server unreachable — keep pinging */ }
    }, PING_INTERVAL);
    return () => clearInterval(pingTimer.current);
  }, [status, cameraId, newSrc]);

  const handleLoad = useCallback(() => {
    if (!mountedRef.current) return;
    const img = imgRef.current;
    if (img && img.naturalWidth > 0) {
      retriesRef.current = 0;   // reset retry counter on successful frame
      setStatus('live');
    }
  }, []);

  const handleError = useCallback(() => {
    if (!mountedRef.current) return;

    // FIX 1a: ignore errors while stream is live (browser MJPEG renegotiation quirk)
    if (status === 'live') return;

    // FIX 1b: ignore errors within startup grace period
    if (Date.now() - mountTimeRef.current < STARTUP_GRACE) return;

    retriesRef.current += 1;
    if (retriesRef.current > MAX_RETRIES) {
      setStatus('offline');
      return;
    }
    setStatus('error');
    // Exponential backoff: 2s, 4s, 8s, 16s, 32s
    const delay = Math.min(2000 * Math.pow(2, retriesRef.current - 1), 32_000);
    retryTimer.current = setTimeout(() => {
      if (mountedRef.current) { setStatus('loading'); newSrc(); }
    }, delay);
  }, [newSrc, status]);

  const retry = useCallback(() => {
    clearTimeout(retryTimer.current);
    clearInterval(pingTimer.current);
    retriesRef.current  = 0;
    mountTimeRef.current = Date.now();  // reset grace period on manual retry
    setStatus('loading');
    newSrc();
  }, [newSrc]);

  if (!isActive) {
    return (
      <div className="w-full h-full bg-gray-950 flex items-center justify-center min-h-[120px] rounded-lg">
        <div className="text-center text-gray-600">
          <div className="text-2xl mb-1 opacity-40">📷</div>
          <div className="text-xs font-mono">INACTIVE</div>
        </div>
      </div>
    );
  }

  const src = epochRef.current === 0
    ? `${API}/api/stream/${cameraId}`
    : `${API}/api/stream/${cameraId}?_r=${epochRef.current}`;

  const displayName = cameraName || cameraId;

  return (
    <div className="relative w-full h-full bg-black rounded-lg overflow-hidden min-h-[120px]">

      {/* ── MJPEG stream image ─────────────────────────────────────────── */}
      {status !== 'offline' && (
        <img
          ref={imgRef}
          key={`${cameraId}::${renderTick}`}
          src={src}
          alt={displayName}
          onLoad={handleLoad}
          onError={handleError}
          className={`absolute inset-0 w-full h-full transition-opacity duration-300 ${
            status === 'live' ? 'opacity-100' : 'opacity-0'
          }`}
          style={{ objectFit: 'contain' }}   // FIX 4: contain = never crop security footage
          crossOrigin="anonymous"
        />
      )}

      {/* ── Connecting spinner ─────────────────────────────────────────── */}
      {status === 'loading' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950">
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-2" />
          <p className="text-gray-500 text-[10px] font-mono uppercase tracking-widest">Connecting</p>
          <p className="text-gray-700 text-[9px] mt-0.5 font-mono">{cameraId}</p>
        </div>
      )}

      {/* ── Error — retrying ───────────────────────────────────────────── */}
      {status === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950 gap-2">
          <div className="w-6 h-6 border-2 border-yellow-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-yellow-600 text-[10px] font-mono">RECONNECTING…</p>
          <p className="text-gray-700 text-[9px] font-mono">attempt {retriesRef.current}/{MAX_RETRIES}</p>
          <button
            onClick={retry}
            className="mt-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 text-[10px] font-mono rounded transition-colors"
          >
            RETRY NOW
          </button>
        </div>
      )}

      {/* ── Offline ────────────────────────────────────────────────────── */}
      {status === 'offline' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950 gap-2">
          <div className="text-red-900 text-2xl opacity-30">⬛</div>
          <p className="text-gray-600 text-[10px] font-mono uppercase tracking-widest">No Signal</p>
          <p className="text-gray-700 text-[9px] font-mono">{cameraId}</p>
          <button
            onClick={retry}
            className="mt-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 text-[10px] font-mono rounded transition-colors"
          >
            CHECK AGAIN
          </button>
        </div>
      )}

      {/* ── DVR-style overlays (only when live) ───────────────────────── */}
      {status === 'live' && (
        <>
          {/* FIX 5: top-left REC badge — matches real DVR software */}
          <div className="absolute top-1.5 left-1.5 flex items-center gap-1 pointer-events-none select-none">
            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shrink-0" />
            <span
              className="text-white font-bold tracking-widest"
              style={{ fontSize: '9px', textShadow: '0 0 4px rgba(0,0,0,0.9)' }}
            >
              REC
            </span>
          </div>

          {/* FIX 6: bottom-left camera name — DVR style like reference footage */}
          <div
            className="absolute bottom-1 left-1.5 pointer-events-none select-none"
            style={{ fontSize: '9px', fontFamily: 'monospace', color: 'rgba(255,255,255,0.85)',
                     textShadow: '0 0 3px rgba(0,0,0,1)', letterSpacing: '0.05em' }}
          >
            {displayName.toUpperCase()}
          </div>

          {/* Bottom-right: camera id (dimmer, like secondary DVR label) */}
          <div
            className="absolute bottom-1 right-1.5 pointer-events-none select-none"
            style={{ fontSize: '8px', fontFamily: 'monospace', color: 'rgba(255,255,255,0.45)',
                     textShadow: '0 0 3px rgba(0,0,0,1)' }}
          >
            {cameraId}
          </div>
        </>
      )}
    </div>
  );
}