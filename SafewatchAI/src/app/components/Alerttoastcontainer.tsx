/**
 * SafeWatch AI — React Alert Popup System
 *
 * WHAT THIS FIXES:
 *   Correctly handles WebSocket type='alert' messages and renders
 *   stacked, auto-dismissing toast notifications.
 *
 * COMPONENTS:
 *   <AlertToastContainer wsUrl="ws://localhost:8000/ws/live" />
 *
 * HOOK (if you manage your own WebSocket):
 *   const { alerts } = useAlertSocket(existingWs)
 *
 * DROP-IN USAGE — add one line to your App.jsx / layout:
 *   import AlertToastContainer from './AlertToastContainer'
 *   // inside <body>/<App>:
 *   <AlertToastContainer wsUrl="ws://localhost:8000/ws/live" />
 *
 * If your app already has a WebSocket (wsRef.current), pass it instead:
 *   <AlertToastContainer ws={wsRef.current} />
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ── Colour / level config ───────────────────────────────────────────────────
const LEVEL = {
  1: { label: "CRITICAL", color: "#ff4444", bg: "rgba(255,44,44,0.15)", border: "rgba(255,44,44,0.35)", dismiss: 10000 },
  2: { label: "HIGH",     color: "#ffaa00", bg: "rgba(255,170,0,0.12)", border: "rgba(255,170,0,0.35)", dismiss: 8000  },
  3: { label: "MEDIUM",   color: "#00e5ff", bg: "rgba(0,229,255,0.10)", border: "rgba(0,229,255,0.35)", dismiss: 6000  },
};

// ── Audio chime ─────────────────────────────────────────────────────────────
function chime(level) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.value = level === 1 ? 880 : level === 2 ? 660 : 440;
    gain.gain.setValueAtTime(0.10, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  } catch (_) {}
}

// ── Single Toast ─────────────────────────────────────────────────────────────
function AlertToast({ alert, onDismiss }) {
  const level   = alert.alert_level || 3;
  const cfg     = LEVEL[level] || LEVEL[3];
  const action  = (alert.action_detected || alert.alert_type || "Unknown").replace(/_/g, " ").toUpperCase();
  const camera  = alert.camera_name || alert.camera_id || "Unknown Camera";
  const zone    = alert.zone ? `Zone ${alert.zone}` : null;
  const time    = new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const [progress, setProgress] = useState(100);
  const [leaving, setLeaving]   = useState(false);

  const handleDismiss = useCallback(() => {
    setLeaving(true);
    setTimeout(onDismiss, 250);
  }, [onDismiss]);

  // Auto-dismiss countdown
  useEffect(() => {
    const duration = cfg.dismiss;
    const interval = 80;
    let elapsed = 0;
    const ticker = setInterval(() => {
      elapsed += interval;
      setProgress(Math.max(0, 100 - (elapsed / duration) * 100));
      if (elapsed >= duration) { clearInterval(ticker); handleDismiss(); }
    }, interval);
    return () => clearInterval(ticker);
  }, [cfg.dismiss, handleDismiss]);

  return (
    <div
      style={{
        background: "#0c0c10",
        border: `1px solid ${cfg.border}`,
        borderRadius: 10,
        overflow: "hidden",
        boxShadow: `0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04), 0 0 20px ${cfg.color}20`,
        fontFamily: "'Courier New', monospace",
        transform: leaving ? "translateX(110%)" : "translateX(0)",
        opacity: leaving ? 0 : 1,
        transition: "transform 0.25s ease-in, opacity 0.25s ease-in",
        pointerEvents: "all",
      }}
    >
      {/* Top stripe */}
      <div style={{ height: 3, background: `linear-gradient(90deg, ${cfg.color}, ${cfg.color}88)` }} />

      {/* Body */}
      <div style={{ display: "flex", gap: 12, padding: "12px 14px" }}>
        {/* Thumbnail */}
        {alert.snapshot_url ? (
          <img
            src={alert.snapshot_url}
            alt="snapshot"
            style={{ width: 72, height: 72, borderRadius: 6, objectFit: "cover", flexShrink: 0, border: "1px solid rgba(255,255,255,0.1)" }}
            onError={(e) => { e.target.style.display = "none"; }}
          />
        ) : (
          <div style={{ width: 72, height: 72, borderRadius: 6, flexShrink: 0, background: "#1a1a24", border: "1px solid rgba(255,255,255,0.1)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28 }}>
            {level === 1 ? "🚨" : level === 2 ? "⚠️" : "🔵"}
          </div>
        )}

        {/* Text info */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Badge */}
          <div style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10, fontWeight: 700, letterSpacing: "1.5px", padding: "3px 8px", borderRadius: 4, marginBottom: 6, background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: cfg.color, display: "inline-block", animation: "sw-pulse 1.2s ease-in-out infinite" }} />
            {cfg.label}
          </div>
          {/* Action */}
          <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", letterSpacing: "0.5px", marginBottom: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {action}
          </div>
          {/* Meta */}
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.6 }}>
            📷 {camera}&nbsp;&nbsp;{zone && <>📍 {zone}&nbsp;&nbsp;</>}🕐 {time}
          </div>
        </div>
      </div>

      {/* Footer — timer + dismiss */}
      <div style={{ display: "flex", alignItems: "center", padding: "0 14px 10px", gap: 10 }}>
        <div style={{ flex: 1, height: 2, background: "rgba(255,255,255,0.08)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${progress}%`, background: cfg.color, borderRadius: 2, transition: "width 0.08s linear" }} />
        </div>
        <button onClick={handleDismiss} style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(255,255,255,0.3)", fontSize: 16, padding: "0 2px", lineHeight: 1 }}
          onMouseEnter={(e) => (e.target.style.color = "rgba(255,255,255,0.8)")}
          onMouseLeave={(e) => (e.target.style.color = "rgba(255,255,255,0.3)")}>
          ✕
        </button>
      </div>
    </div>
  );
}

// ── Hook: attach to existing WebSocket ──────────────────────────────────────
export function useAlertSocket(ws) {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    if (!ws) return;

    function handleMessage(event) {
      try {
        const data = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
        if (data?.type === "alert" && data.alert) {
          const newAlert = { ...data.alert, _id: Date.now() + Math.random() };
          chime(newAlert.alert_level || 3);
          setAlerts((prev) => [newAlert, ...prev].slice(0, 5)); // max 5
        }
      } catch (_) {}
    }

    ws.addEventListener("message", handleMessage);
    return () => ws.removeEventListener("message", handleMessage);
  }, [ws]);

  const dismiss = useCallback((id) => {
    setAlerts((prev) => prev.filter((a) => a._id !== id));
  }, []);

  return { alerts, dismiss };
}

// ── Container component ──────────────────────────────────────────────────────
export default function AlertToastContainer({ wsUrl, ws: externalWs }) {
  const [alerts, setAlerts]  = useState([]);
  const wsRef = useRef(null);

  // If caller passes a wsUrl, create and manage the WebSocket here
  useEffect(() => {
    if (externalWs || !wsUrl) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onopen  = () => console.log("[SafeWatch] WS connected");
    ws.onclose = () => console.log("[SafeWatch] WS closed");
    return () => { ws.close(); };
  }, [wsUrl, externalWs]);

  // Attach message handler to whichever WebSocket we're using
  useEffect(() => {
    const ws = externalWs || wsRef.current;
    if (!ws) return;

    function onMessage(event) {
      try {
        const data = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
        if (data?.type === "alert" && data.alert) {
          const newAlert = { ...data.alert, _id: `${Date.now()}_${Math.random()}` };
          chime(newAlert.alert_level || 3);
          setAlerts((prev) => [newAlert, ...prev].slice(0, 5));
        }
      } catch (_) {}
    }

    ws.addEventListener("message", onMessage);
    return () => ws.removeEventListener("message", onMessage);
  }, [externalWs]);

  const dismiss = useCallback((id) => {
    setAlerts((prev) => prev.filter((a) => a._id !== id));
  }, []);

  return (
    <>
      {/* Pulse keyframe — injected once */}
      <style>{`@keyframes sw-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }`}</style>

      <div style={{
        position: "fixed",
        top: 20,
        right: 20,
        zIndex: 99999,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        maxWidth: 380,
        width: "100%",
        pointerEvents: "none",
      }}>
        {alerts.map((alert) => (
          <AlertToast
            key={alert._id}
            alert={alert}
            onDismiss={() => dismiss(alert._id)}
          />
        ))}
      </div>
    </>
  );
}