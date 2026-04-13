/**
 * MiniNotificationPopup.tsx — v2 Fixed Build
 *
 * FIX: Known persons (Employee / Owner) no longer appear in the popup.
 *
 * ROOT CAUSE: The popup received `unknownPersons` from Dashboard, but
 * Dashboard's `unknownPersons` state could include WS-pushed persons
 * of category Employee or Owner (handleDetectionFromWS guard only checked
 * for face_image_url, not the classification).
 *
 * FIX APPLIED HERE (belt-and-suspenders):
 *   - Filter incoming unknownPersons to only show category === 'Unknown'
 *   - Additionally skip any person whose name is NOT 'Unknown Person' AND
 *     whose recognizedAs field is set (i.e. backend already identified them)
 *
 * The upstream fix (in Dashboard) ensures only Unknown-classified persons
 * are pushed to unknownPersons at all. This component adds a local guard
 * so even if something slips through, it won't be shown.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, UserX, AlertTriangle, ChevronDown, ChevronUp, Bell } from 'lucide-react';
import { Person } from './PersonClassifier';
import { SuspiciousActivity } from './SuspiciousActivityAlert';

interface MiniNotif {
  id: string;
  type: 'unknown' | 'alert';
  person?: Person;
  activity?: SuspiciousActivity;
  arrivedAt: number;
}

interface MiniNotificationPopupProps {
  unknownPersons: Person[];
  suspiciousActivities: SuspiciousActivity[];
  resolvedAlertIds: Set<string>;
  onClassifyPerson: (person: Person) => void;
  onViewActivity: (activity: SuspiciousActivity) => void;
}

const SEVERITY_COLOR: Record<string, string> = {
  critical: '#ff3333',
  high:     '#ff8800',
  medium:   '#ffcc00',
  low:      '#3399ff',
};

function PersonThumbnail({ imageData }: { imageData?: string }) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!imageData) { setFailed(true); return; }
    if (imageData.startsWith('http')) { setSrc(imageData); return; }
    const filename = imageData.split(/[/\\]/).pop();
    if (filename) setSrc(`http://localhost:8000/media/faces/unknown/${filename}`);
    else setFailed(true);
  }, [imageData]);

  if (failed || !src) {
    return (
      <div style={{
        width: 40, height: 40, borderRadius: 8,
        background: 'rgba(255,165,0,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>
        <UserX size={20} color="#ff9900" />
      </div>
    );
  }
  return (
    <img
      src={src}
      alt="Unknown"
      onError={() => setFailed(true)}
      style={{
        width: 40, height: 40, borderRadius: 8,
        objectFit: 'cover', flexShrink: 0,
        border: '1px solid rgba(255,153,0,0.4)',
      }}
    />
  );
}

export function MiniNotificationPopup({
  unknownPersons: _unknownPersonsRaw,
  suspiciousActivities,
  resolvedAlertIds,
  onClassifyPerson,
  onViewActivity,
}: MiniNotificationPopupProps) {
  const [notifs, setNotifs] = useState<MiniNotif[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const seenRef = useRef(new Set<string>());
  const collapseTimer = useRef<ReturnType<typeof setTimeout>>();
  const lastAlertCountRef = useRef(0);

  // ── FIX: Only show genuinely unknown persons ───────────────────────────────
  // Filter out Employee/Owner and already-recognized persons
  const unknownPersons = _unknownPersonsRaw.filter(p =>
    p.category === 'Unknown' &&
    !p.recognizedAs   // skip auto-recognized (backend already matched them)
  );

  // Auto-collapse after 8s of no hover
  const resetCollapseTimer = useCallback(() => {
    clearTimeout(collapseTimer.current);
    setCollapsed(false);
    if (!isHovered) {
      collapseTimer.current = setTimeout(() => setCollapsed(true), 8000);
    }
  }, [isHovered]);

  useEffect(() => {
    seenRef.current.clear();
    lastAlertCountRef.current = 0;
  }, []);

  // Track new unknown persons — only genuinely unknown, category === 'Unknown'
  useEffect(() => {
    const recent = unknownPersons.slice(0, 5);
    let added = false;
    for (const p of recent) {
      const key = `unknown-${p.id}`;
      if (seenRef.current.has(key)) continue;
      seenRef.current.add(key);
      const notif: MiniNotif = {
        id: key, type: 'unknown', person: p, arrivedAt: Date.now(),
      };
      setNotifs(prev => [notif, ...prev].slice(0, 5));
      added = true;
    }
    if (added) resetCollapseTimer();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unknownPersons.length, unknownPersons[0]?.id, resetCollapseTimer]);

  // Track new alerts
  useEffect(() => {
    const active = suspiciousActivities.filter(a => !resolvedAlertIds.has(a.id));
    if (active.length <= lastAlertCountRef.current && lastAlertCountRef.current > 0) {
      return;
    }
    let added = false;
    const newAlerts = active.slice(0, active.length - lastAlertCountRef.current);
    for (const a of newAlerts.reverse()) {
      const key = `alert-${a.id}`;
      if (seenRef.current.has(key)) continue;
      seenRef.current.add(key);
      const notif: MiniNotif = {
        id: key, type: 'alert', activity: a, arrivedAt: Date.now(),
      };
      setNotifs(prev => [notif, ...prev].slice(0, 5));
      added = true;
    }
    lastAlertCountRef.current = active.length;
    if (added) resetCollapseTimer();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [suspiciousActivities.length, resolvedAlertIds.size, resetCollapseTimer]);

  // Remove notifs for resolved alerts
  useEffect(() => {
    setNotifs(prev =>
      prev.filter(n => !(n.type === 'alert' && n.activity && resolvedAlertIds.has(n.activity.id)))
    );
  }, [resolvedAlertIds]);

  const dismiss = (id: string) => setNotifs(prev => prev.filter(n => n.id !== id));
  const dismissAll = () => { setNotifs([]); setCollapsed(true); };

  const visible = notifs.slice(0, 3);
  if (visible.length === 0) return null;

  return (
    <div
      style={{
        position: 'fixed', bottom: 20, right: 20, zIndex: 9000,
        display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6,
        pointerEvents: 'none',
      }}
      onMouseEnter={() => { setIsHovered(true); setCollapsed(false); clearTimeout(collapseTimer.current); }}
      onMouseLeave={() => { setIsHovered(false); resetCollapseTimer(); }}
    >
      {/* Collapsed badge */}
      <AnimatePresence>
        {collapsed && (
          <motion.button
            key="badge"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={() => setCollapsed(false)}
            style={{
              pointerEvents: 'all',
              background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
              border: '1px solid rgba(255,100,0,0.4)',
              borderRadius: 20, padding: '6px 12px',
              display: 'flex', alignItems: 'center', gap: 6,
              cursor: 'pointer', color: '#fff',
              boxShadow: '0 4px 20px rgba(0,0,0,0.5), 0 0 12px rgba(255,80,0,0.2)',
            }}
          >
            <Bell size={14} color="#ff6600" />
            <span style={{ fontSize: 12, fontWeight: 700, fontFamily: 'monospace' }}>
              {visible.length} notification{visible.length !== 1 ? 's' : ''}
            </span>
            <ChevronUp size={12} color="#aaa" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Expanded cards */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            key="expanded"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            style={{
              pointerEvents: 'all',
              display: 'flex', flexDirection: 'column', gap: 6,
              alignItems: 'flex-end',
            }}
          >
            {/* Header with collapse + dismiss-all */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 8px',
              background: 'rgba(15,15,25,0.85)',
              borderRadius: 8, border: '1px solid rgba(255,255,255,0.06)',
            }}>
              <Bell size={12} color="#888" />
              <span style={{ fontSize: 10, color: '#888', fontFamily: 'monospace', fontWeight: 600 }}>
                LIVE NOTIFICATIONS
              </span>
              <button
                onClick={dismissAll}
                style={{ background: 'none', border: 'none', cursor: 'pointer',
                         color: '#555', fontSize: 10, marginLeft: 4 }}
              >
                Clear all
              </button>
              <button
                onClick={() => setCollapsed(true)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#555' }}
              >
                <ChevronDown size={12} />
              </button>
            </div>

            {/* Notification cards */}
            {visible.map((notif, i) => (
              <NotifCard
                key={notif.id}
                notif={notif}
                isNewest={i === 0}
                onDismiss={dismiss}
                onClassify={onClassifyPerson}
                onView={onViewActivity}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Individual card ──────────────────────────────────────────────────────────
function NotifCard({
  notif, isNewest, onDismiss, onClassify, onView,
}: {
  notif: MiniNotif;
  isNewest: boolean;
  onDismiss: (id: string) => void;
  onClassify: (p: Person) => void;
  onView: (a: SuspiciousActivity) => void;
}) {
  const isUnknown = notif.type === 'unknown';
  const accentColor = isUnknown
    ? '#ff9900'
    : SEVERITY_COLOR[notif.activity?.severity ?? 'low'] ?? '#3399ff';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 60, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 60, scale: 0.9 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      style={{
        width: 280,
        background: 'linear-gradient(135deg, #0e0e18, #141425)',
        border: `1px solid ${accentColor}44`,
        borderLeft: `3px solid ${accentColor}`,
        borderRadius: 10,
        overflow: 'hidden',
        boxShadow: `0 8px 32px rgba(0,0,0,0.6), 0 0 16px ${accentColor}18`,
        fontFamily: "'Courier New', monospace",
        position: 'relative',
      }}
    >
      {isNewest && (
        <motion.div
          initial={{ opacity: 1 }}
          animate={{ opacity: 0 }}
          transition={{ delay: 3, duration: 1 }}
          style={{
            position: 'absolute', top: 8, right: 28,
            width: 7, height: 7, borderRadius: '50%',
            background: accentColor,
            boxShadow: `0 0 8px ${accentColor}`,
          }}
        />
      )}

      <div style={{ height: 2, background: `linear-gradient(90deg, ${accentColor}, transparent)` }} />

      <div style={{ padding: '10px 12px' }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {isUnknown
              ? <UserX size={14} color={accentColor} />
              : <AlertTriangle size={14} color={accentColor} />}
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: '1.5px',
              color: accentColor, textTransform: 'uppercase',
            }}>
              {isUnknown ? 'Unknown Person' : `${notif.activity?.severity ?? 'alert'} alert`}
            </span>
          </div>
          <button
            onClick={() => onDismiss(notif.id)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'rgba(255,255,255,0.3)', fontSize: 14, lineHeight: 1, padding: 0,
            }}
          >×</button>
        </div>

        {/* Body */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {isUnknown && notif.person && (
            <PersonThumbnail imageData={notif.person.imageData} />
          )}
          {!isUnknown && (
            <div style={{
              width: 40, height: 40, borderRadius: 8, flexShrink: 0,
              background: `${accentColor}18`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ fontSize: 20 }}>
                {notif.activity?.severity === 'critical' ? '🚨'
                 : notif.activity?.severity === 'high'   ? '⚠️' : '⚡'}
              </span>
            </div>
          )}

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 12, fontWeight: 700, color: '#fff',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              marginBottom: 2,
            }}>
              {isUnknown
                ? (notif.person?.name || 'Unknown Person')
                : (notif.activity?.type ?? '').replace(/_/g, ' ').toUpperCase()}
            </div>
            <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', lineHeight: 1.5 }}>
              {isUnknown
                ? `Detected • ${notif.person?.timestamp.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`
                : `📷 ${notif.activity?.cameraName ?? 'Camera'}`}
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
          {isUnknown && notif.person ? (
            <button
              onClick={() => onClassify(notif.person!)}
              style={{
                flex: 1, padding: '5px 0', borderRadius: 6, border: 'none',
                background: `linear-gradient(90deg, ${accentColor}cc, ${accentColor}88)`,
                color: '#000', fontSize: 10, fontWeight: 700, cursor: 'pointer',
                letterSpacing: '0.5px',
              }}
            >
              CLASSIFY
            </button>
          ) : (
            <button
              onClick={() => onView(notif.activity!)}
              style={{
                flex: 1, padding: '5px 0', borderRadius: 6, border: 'none',
                background: `linear-gradient(90deg, ${accentColor}cc, ${accentColor}66)`,
                color: '#fff', fontSize: 10, fontWeight: 700, cursor: 'pointer',
                letterSpacing: '0.5px',
              }}
            >
              VIEW ALERT
            </button>
          )}
          <button
            onClick={() => onDismiss(notif.id)}
            style={{
              padding: '5px 10px', borderRadius: 6,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'transparent', color: 'rgba(255,255,255,0.4)',
              fontSize: 10, cursor: 'pointer',
            }}
          >
            Dismiss
          </button>
        </div>
      </div>
    </motion.div>
  );
}