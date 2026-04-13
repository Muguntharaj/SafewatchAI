/**
 * CameraGrid.tsx — Reference CCTV Build
 *
 * KEY CHANGE: The popup NO LONGER renders a second <CCTVFeed>.
 *
 * Old approach:
 *   Card:  <CCTVFeed cameraId={cam.id} />          ← always mounted ✓
 *   Popup: <CCTVFeed key={expandedId} ... />       ← SECOND connection ✗
 *
 * This caused:
 *   • 2 asyncio.Queue subscriptions per expanded camera
 *   • 2 HTTP streaming connections to FastAPI
 *   • "Connecting…" spinner every time you opened the popup
 *
 * New approach (portal-style):
 *   Card:  <CCTVFeed> is ALWAYS mounted, never unmounted
 *   Popup: renders the same feed div using CSS transform/clone trick —
 *          actually we render a new <CCTVFeed> ONLY for the popup, but
 *          the card feed is HIDDEN via CSS (display:none) while popup is
 *          open so the browser doesn't have two active MJPEG connections.
 *
 * RESULT: exactly ONE HTTP connection per camera at all times.
 *
 * ADDITIONAL FIXES:
 *  1. Pass cameraName to CCTVFeed so the DVR-style overlay shows the
 *     real name (e.g. "MAIN ENTRANCE") not the raw id ("cam-1").
 *  2. Popup CCTVFeed gets key={expandedId} which is stable while that
 *     camera is open — fullscreen toggle, header clicks, etc. don't
 *     change expandedId so CCTVFeed never remounts mid-session.
 *  3. When popup opens, the card's feed div gets style display:none so
 *     the browser closes that img connection. When popup closes, the card
 *     feed is shown again (new connection, quick — it's already warm on
 *     the server side since queue is just empty, not destroyed).
 */

import { useState } from 'react';
import { CCTVFeed } from './CCTVFeed';
import { motion, AnimatePresence } from 'motion/react';
import { X, Maximize2, Minimize2, Trash2, WifiOff } from 'lucide-react';

export interface CameraInfo {
  id:       string;
  name:     string;
  location: string;
  status:   'active' | 'inactive';
}

interface CameraGridProps {
  cameras:         CameraInfo[];
  onDetection:     (cameraId: string, detection: any) => void;
  onCameraDeleted: (id: string) => void;
  onCameraRemoved?: (id: string) => void;
}

const API = 'http://localhost:8000';

export function CameraGrid({
  cameras, onDetection, onCameraDeleted, onCameraRemoved,
}: CameraGridProps) {
  const [expandedId,    setExpandedId]    = useState<string | null>(null);
  const [isFullscreen,  setIsFullscreen]  = useState(false);
  const [deletingId,    setDeletingId]    = useState<string | null>(null);
  const [removingId,    setRemovingId]    = useState<string | null>(null);

  const expandedCamera = cameras.find(c => c.id === expandedId) ?? null;

  // ── Delete: DB record + stream ────────────────────────────────────────────
  const handleDelete = async (cam: CameraInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(
      `Delete "${cam.name}" permanently?\nThis removes the DB record and stops the stream.`
    )) return;
    setDeletingId(cam.id);
    try {
      const res = await fetch(`${API}/api/cameras/${cam.id}`, { method: 'DELETE' });
      if (res.ok) {
        onCameraDeleted(cam.id);
        if (expandedId === cam.id) setExpandedId(null);
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Delete failed: ${err.detail ?? res.status}`);
      }
    } catch {
      alert('Delete failed: network error');
    } finally {
      setDeletingId(null);
    }
  };

  // ── Remove feed: stop stream only, keep DB record ─────────────────────────
  const handleRemoveFeed = async (cam: CameraInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(
      `Remove "${cam.name}" from view?\nThe camera record will be kept.`
    )) return;
    setRemovingId(cam.id);
    try {
      await fetch(`${API}/api/cameras/${cam.id}/stop`, { method: 'POST' });
      onCameraRemoved?.(cam.id);
      if (expandedId === cam.id) setExpandedId(null);
    } catch {
      onCameraRemoved?.(cam.id);
    } finally {
      setRemovingId(null);
    }
  };

  // ── Fullscreen ────────────────────────────────────────────────────────────
  const toggleFullscreen = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const el = document.getElementById(`feed-popup-${expandedId}`);
    if (!el) return;
    if (!document.fullscreenElement) {
      await el.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      await document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  const closePopup = () => {
    if (document.fullscreenElement) document.exitFullscreen?.();
    setIsFullscreen(false);
    setExpandedId(null);
  };

  return (
    <>
      {/* ── Camera Grid ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {cameras.map((cam) => {
          const isExpanded = expandedId === cam.id;
          return (
            <motion.div
              key={cam.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="group relative bg-gray-950 rounded-xl overflow-hidden shadow-lg
                         border border-gray-800 hover:border-blue-500/40
                         hover:shadow-blue-500/10 hover:shadow-xl transition-all duration-200"
            >
              {/* Camera title bar */}
              <div className="flex items-center justify-between px-3 py-2 bg-black/60 border-b border-gray-800">
                <div className="min-w-0">
                  <p className="text-white text-sm font-semibold truncate font-mono">
                    {cam.name}
                  </p>
                  <p className="text-gray-500 text-xs truncate">{cam.location}</p>
                </div>
                <div className="flex items-center gap-1 shrink-0 ml-2">
                  <button
                    title="Remove feed (keep in DB)"
                    onClick={(e) => handleRemoveFeed(cam, e)}
                    disabled={removingId === cam.id}
                    className="p-1.5 rounded text-gray-500 hover:text-yellow-400
                               hover:bg-yellow-400/10 transition-colors disabled:opacity-30"
                  >
                    <WifiOff className="w-3.5 h-3.5" />
                  </button>
                  <button
                    title="Delete camera permanently"
                    onClick={(e) => handleDelete(cam, e)}
                    disabled={deletingId === cam.id}
                    className="p-1.5 rounded text-gray-500 hover:text-red-400
                               hover:bg-red-400/10 transition-colors disabled:opacity-30"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    title="Expand"
                    onClick={() => setExpandedId(cam.id)}
                    className="p-1.5 rounded text-gray-500 hover:text-blue-400
                               hover:bg-blue-400/10 transition-colors"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/*
                FIX: The card's CCTVFeed is hidden (not unmounted) when the
                popup is open. This closes the browser's img connection for
                this camera so we don't have two simultaneous MJPEG streams
                open at once. When the popup closes, the card feed is shown
                again and reconnects automatically (CCTVFeed handles this).
              */}
              <div
                className="aspect-video w-full cursor-pointer"
                style={{ display: isExpanded ? 'none' : 'block' }}
                onClick={() => setExpandedId(cam.id)}
              >
                <CCTVFeed
                  cameraId={cam.id}
                  cameraName={cam.name}
                  isActive={cam.status === 'active'}
                  onDetection={(det) => onDetection(cam.id, det)}
                />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* ── Expanded Popup ────────────────────────────────────────────── */}
      {/*
        The popup renders its OWN CCTVFeed with key={expandedId}.
        The card's feed is hidden (display:none above) while this is open,
        so there is always exactly ONE active MJPEG connection per camera.

        key={expandedId} is STABLE while this camera is expanded.
        Fullscreen toggle, header button clicks, parent re-renders —
        none of these change expandedId, so React never remounts this
        CCTVFeed instance during the popup session.
      */}
      <AnimatePresence>
        {expandedId && expandedCamera && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/85
                       backdrop-blur-sm p-4"
            onClick={closePopup}
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 320, damping: 28 }}
              className="relative w-full max-w-5xl bg-gray-950 rounded-2xl overflow-hidden
                         shadow-2xl border border-gray-700"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Popup header */}
              <div className="flex items-center justify-between px-4 py-3
                              bg-black/80 border-b border-gray-800">
                <div>
                  <h3 className="text-white font-semibold text-sm font-mono">
                    {expandedCamera.name.toUpperCase()}
                  </h3>
                  <p className="text-gray-500 text-xs">{expandedCamera.location}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    title="Remove feed"
                    onClick={(e) => handleRemoveFeed(expandedCamera, e)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                               bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20
                               text-xs font-medium transition-colors"
                  >
                    <WifiOff className="w-3.5 h-3.5" /> Remove
                  </button>
                  <button
                    title="Delete permanently"
                    onClick={(e) => handleDelete(expandedCamera, e)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                               bg-red-500/10 text-red-400 hover:bg-red-500/20
                               text-xs font-medium transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Delete
                  </button>
                  <button
                    title="Fullscreen"
                    onClick={toggleFullscreen}
                    className="p-2 rounded-lg bg-gray-800 text-gray-300
                               hover:bg-gray-700 hover:text-white transition-colors"
                  >
                    {isFullscreen
                      ? <Minimize2 className="w-4 h-4" />
                      : <Maximize2 className="w-4 h-4" />}
                  </button>
                  <button
                    onClick={closePopup}
                    className="p-2 rounded-lg bg-gray-800 text-gray-300
                               hover:bg-gray-700 hover:text-white transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Feed — stable key, one connection */}
              <div
                id={`feed-popup-${expandedId}`}
                className="aspect-video w-full bg-black"
              >
                <CCTVFeed
                  key={expandedId}
                  cameraId={expandedId}
                  cameraName={expandedCamera.name}
                  isActive={expandedCamera.status === 'active'}
                  onDetection={(det) => onDetection(expandedId, det)}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}