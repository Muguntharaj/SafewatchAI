/**
 * Dashboard.tsx — SafeWatch AI — Complete Production Build
 *
 * FIXES:
 * FIX-1  fetchStats was hitting /api/stats (non-existent) → now correctly
 *        hits /api/analytics/dashboard and maps all 5 stat fields.
 *        prevStatsRef removed (was undefined, silent crash).
 *
 * FIX-2  handleDetectionFromWS: camera.name crash when camera is undefined.
 *        Added safe placeholder. Buffered early WS detections before cameras load.
 *
 * FIX-3  Known employees/owners NEVER appear in unknown list or notifications.
 *        Hard guard: isGenuinelyUnknown requires category='Unknown' AND no
 *        person.name from backend AND face_image_url present.
 *
 * FIX-4  handleClassify immediately removes classified person from unknownPersons.
 *
 * FIX-5  Alert dedup on WS: prevents duplicate alert entries on re-render.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { CameraGrid, CameraInfo }                    from './CameraGrid';
import { PersonClassifier, Person }                  from './PersonClassifier';
import { TimeLogTable, TimeLog }                     from './TimeLogTable';
import { SuspiciousActivityAlert, SuspiciousActivity } from './SuspiciousActivityAlert';
import { NotificationSlider }                        from './NotificationSlider';
import { MiniNotificationPopup }                     from './MiniNotificationPopup';
import { Header }                                    from './Header';
import { StatsCards }                                from './StatsCards';
import { HomePage }                                  from './HomePage';
import { AboutPage }                                 from './AboutPage';
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from './ui/card';
import { Button } from './ui/button';
import { Badge }  from './ui/badge';
import { AlertTriangle, Camera, Clock, Video, RefreshCw } from 'lucide-react';
import { toast }  from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { motion } from 'motion/react';
import { AddCamera } from './AddCamera';

const API = 'http://localhost:8000';

interface DashboardProps {
  username: string;
  role:     string;
  onLogout: () => void;
}

const getFaceImageUrl = (path?: string | null): string | undefined => {
  if (!path) return undefined;
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const filename = path.split(/[/\\]/).pop();
  if (!filename || filename.includes(':')) return undefined;
  return `${API}/media/faces/unknown/${filename}`;
};

const getAlertClipUrl = (alertId: string, videoPath?: string | null): string => {
  if (videoPath) {
    if (videoPath.startsWith('http://') || videoPath.startsWith('https://')) return videoPath;
    const filename = videoPath.split(/[/\\]/).pop();
    if (filename) return `${API}/media/alerts/${filename}`;
  }
  return `${API}/media/alerts/${alertId}_clip.mp4`;
};

const getSnapshotUrl = (alertId: string, snapshotPath?: string | null): string => {
  if (snapshotPath) {
    if (snapshotPath.startsWith('http://') || snapshotPath.startsWith('https://')) return snapshotPath;
    const filename = snapshotPath.split(/[/\\]/).pop();
    if (filename) return `${API}/media/alerts/${filename}`;
  }
  return `${API}/media/alerts/${alertId}_snapshot.jpg`;
};

export function Dashboard({ username, role, onLogout }: DashboardProps) {
  const [currentView, setCurrentView] = useState<'home' | 'about'>('home');
  const [cameras, setCameras]         = useState<CameraInfo[]>([]);
  const [camerasLoading, setCamerasLoading] = useState(true);
  const [persons, setPersons]                           = useState<Person[]>([]);
  const [unknownPersons, setUnknownPersons]             = useState<Person[]>([]);
  const [timeLogs, setTimeLogs]                         = useState<TimeLog[]>([]);
  const [suspiciousActivities, setSuspiciousActivities] = useState<SuspiciousActivity[]>([]);
  const [selectedPerson, setSelectedPerson]             = useState<Person | null>(null);
  const [selectedActivity, setSelectedActivity]         = useState<SuspiciousActivity | null>(null);
  const [isClassifierOpen, setIsClassifierOpen]         = useState(false);
  const [isActivityAlertOpen, setIsActivityAlertOpen]   = useState(false);
  const [isNotificationSliderOpen, setIsNotificationSliderOpen] = useState(false);
  const [notificationInitialTab, setNotificationInitialTab] =
    useState<'alerts' | 'unknown' | 'recent'>('alerts');
  const [alertResetLoading, setAlertResetLoading] = useState(false);
  const [stats, setStats] = useState({ total: 0, known: 0, unknown: 0, currentlyIn: 0, alerts: 0 });
  const [classifiedPersonIds, setClassifiedPersonIds] = useState<Set<string>>(new Set());
  const [resolvedAlertIds, setResolvedAlertIds]       = useState<Set<string>>(new Set());

  const processedDetectionsRef = useRef(new Set<string>());
  const seenAlertIdsRef        = useRef(new Set<string>());
  const camerasRef             = useRef<CameraInfo[]>([]);
  const unknownPersonsRef      = useRef<Person[]>([]);
  const personsRef             = useRef<Person[]>([]);
  const earlyDetectionBuffer   = useRef<Array<{cameraId: string; detection: any}>>([]);
  const camerasLoadedRef       = useRef(false);

  useEffect(() => { unknownPersonsRef.current = unknownPersons; }, [unknownPersons]);
  useEffect(() => { personsRef.current = persons; }, [persons]);

  // ── 1. Cameras ────────────────────────────────────────────────────────────
  useEffect(() => {
    const load = (isFirst = false) => {
      if (isFirst) setCamerasLoading(true);
      fetch(`${API}/api/cameras/list`)
        .then(r => r.ok ? r.json() : Promise.reject())
        .then((data: any[]) => {
          const arr    = Array.isArray(data) ? data : (data.cameras ?? []);
          const mapped = arr
            .filter((c: any) => c.status !== 'inactive')
            .map((c: any) => ({
              id:       c.camera_id ?? String(c.id),
              name:     c.name,
              location: c.location ?? '',
              status:   c.status === 'active' ? 'active' : 'inactive',
            })) as CameraInfo[];
          setCameras(mapped);
          camerasRef.current = mapped;
          // FIX-2: replay buffered detections once cameras are loaded
          if (!camerasLoadedRef.current) {
            camerasLoadedRef.current = true;
            const buffered = earlyDetectionBuffer.current.splice(0);
            for (const { cameraId, detection } of buffered) {
              handleDetectionFromWS(cameraId, detection);
            }
          }
        })
        .catch(() => {})
        .finally(() => { if (isFirst) setCamerasLoading(false); });
    };
    load(true);
    const id = setInterval(() => load(false), 30_000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 2. WebSocket ──────────────────────────────────────────────────────────
  useEffect(() => {
    const connect = (): WebSocket => {
      const ws = new WebSocket('ws://localhost:8000/ws/live');
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'detections')
            msg.detections.forEach((det: any) => handleDetectionFromWS(msg.camera_id, det));
          if (msg.type === 'alert') {
            const a         = msg.alert;
            const videoClip = a.video_url    ?? getAlertClipUrl(a.alert_id, a.video_path);
            const snapshot  = a.snapshot_url ?? getSnapshotUrl(a.alert_id, a.snapshot_path);
            const activity: SuspiciousActivity = {
              id: a.alert_id, cameraId: String(a.camera_id),
              cameraName: a.camera_name ?? `Camera ${a.camera_id}`,
              type: a.alert_type as any, severity: a.severity as any,
              timestamp: new Date(a.created_at), description: a.description,
              videoClip, snapshot,
            };
            // FIX-5: dedup before adding
            setSuspiciousActivities(prev => {
              if (prev.some(x => x.id === activity.id)) return prev;
              return [activity, ...prev.slice(0, 99)];
            });
            if (!seenAlertIdsRef.current.has(a.alert_id)) {
              seenAlertIdsRef.current.add(a.alert_id);
              const emoji = a.severity === 'critical' ? '🚨' : a.severity === 'high' ? '⚠️' : '⚡';
              toast.error(`${emoji} ${(a.severity ?? '').toUpperCase()} — ${a.alert_type}`, {
                description: `${a.description} · ${activity.cameraName}`, duration: 8000,
                action: { label: 'View Details',
                  onClick: () => { setSelectedActivity(activity); setIsActivityAlertOpen(true); } },
              });
              if (a.alert_level === 1 || a.severity === 'critical') {
                setNotificationInitialTab('alerts');
                setIsNotificationSliderOpen(true);
                setTimeout(() => { setSelectedActivity(activity); setIsActivityAlertOpen(true); }, 300);
              }
            }
            fetchStatsRef.current();
          }
        } catch { }
      };
      ws.onerror = () => console.warn('WS error');
      ws.onclose = () => setTimeout(connect, 3000);
      return ws;
    };
    const ws = connect();
    return () => ws.close();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── 3. Stats — FIX-1 ────────────────────────────────────────────────────
  const fetchStatsRef = useRef<() => Promise<void>>(() => Promise.resolve());

  const fetchStats = useCallback(async () => {
    try {
      // FIX-1: correct endpoint (was /api/stats which doesn't exist)
      const r = await fetch(`${API}/api/analytics/dashboard?period=today`);
      if (!r.ok) return;
      const d = await r.json();
      setStats({
        total:       d.total_detections ?? 0,
        known:       d.known_persons     ?? 0,
        unknown:     d.unknown_persons   ?? 0,
        currentlyIn: d.currently_in      ?? 0,
        alerts:      d.total_alerts      ?? 0,
      });
    } catch { }
  }, []);
  fetchStatsRef.current = fetchStats;

  // ── 4. Unknown persons ────────────────────────────────────────────────────
  const fetchUnknown = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/faces/unknown/list?limit=50`);
      if (!r.ok) return;
      const data  = await r.json();
      const items = Array.isArray(data) ? data : (data.value ?? data.data ?? []);
      const mapped: Person[] = items.map((u: any) => ({
        id:         String(u.id),
        category:   'Unknown' as const,
        confidence: u.confidence ?? 1,
        imageData:  u.face_image_url ?? getFaceImageUrl(u.face_image_path),
        timestamp:  new Date((u.detection_time ?? u.timestamp ?? new Date().toISOString())
                      + (String(u.detection_time ?? '').endsWith('Z') ? '' : 'Z')),
        name:       u.suggested_name ?? u.name ?? undefined,
        cameraId:   u.camera_id ? String(u.camera_id) : undefined,
      }));
      // FIX-3: only truly unknown persons
      const onlyUnknown = mapped.filter(p => p.category === 'Unknown');
      setUnknownPersons(prev => {
        const existingIds = new Set(prev.map(u => u.id));
        const newItems    = onlyUnknown.filter(u => !existingIds.has(u.id));
        if (newItems.length === 0) return prev;
        return [...newItems, ...prev].slice(0, 50);
      });
      unknownPersonsRef.current = onlyUnknown;
    } catch { }
  }, []);

  // ── 5. Time logs ─────────────────────────────────────────────────────────
  const fetchLogs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/detections/recent?limit=100`);
      if (!r.ok) return;
      const data     = await r.json();
      const unknowns = unknownPersonsRef.current;
      const known    = personsRef.current;

      const logs: TimeLog[] = data.map((d: any) => {
        const ts   = new Date(d.timestamp + (d.timestamp.endsWith('Z') ? '' : 'Z'));
        const pid  = d.person_id ? String(d.person_id) : (d.detection_uuid ?? String(d.id));
        const name = d.person_name
          || (d.action && d.action !== 'normal' ? d.action : 'Unknown Person');
        const cat: TimeLog['category'] =
          d.person_classification === 'employee' ? 'Employee'
          : d.person_classification === 'owner'  ? 'Owner' : 'Unknown';

        let imageData: string | undefined = d.face_image_url ?? undefined;
        if (!imageData) {
          const byId = unknowns.find(u => u.id === pid);
          if (byId?.imageData) imageData = byId.imageData;
        }
        if (!imageData) {
          const byKnown = known.find(p => p.id === pid);
          if (byKnown?.imageData) imageData = byKnown.imageData;
        }
        if (!imageData && name !== 'Unknown Person') {
          const byName = [...known, ...unknowns].find(p => p.name === name && p.imageData);
          if (byName?.imageData) imageData = byName.imageData;
        }
        if (!imageData) {
          const closest = unknowns.reduce<Person | null>((best, u) => {
            const diff = Math.abs(u.timestamp.getTime() - ts.getTime());
            if (diff > 90_000) return best;
            if (!best) return u;
            return diff < Math.abs(best.timestamp.getTime() - ts.getTime()) ? u : best;
          }, null);
          if (closest?.imageData) imageData = closest.imageData;
        }

        return {
          id: String(d.id), personId: pid, personName: name,
          category: cat, inTime: ts, status: 'In' as const,
          cameraId: String(d.camera_id),
          cameraName: d.camera_label || `Camera ${d.camera_id}`,
          imageData,
        };
      });
      setTimeLogs(logs);
    } catch { }
  }, []);

  // ── 6. Alerts ─────────────────────────────────────────────────────────────
  const fetchAlerts = useCallback(async () => {
    try {
      const today = new Date().toISOString().split('T')[0];
      const r     = await fetch(`${API}/api/alerts/list?limit=50&date=${today}`);
      if (!r.ok) return;
      const data  = await r.json();
      const mapped: SuspiciousActivity[] = data
        .filter((a: any) => !resolvedAlertIds.has(a.alert_id))
        .map((a: any) => ({
          id: a.alert_id, cameraId: String(a.camera_id ?? ''),
          cameraName: a.camera_name ?? `Camera ${a.camera_id ?? '?'}`,
          type: a.alert_type as any, severity: a.severity as any,
          timestamp:   new Date(a.created_at + (a.created_at.endsWith('Z') ? '' : 'Z')),
          description: a.description,
          videoClip:   a.video_url    ?? getAlertClipUrl(a.alert_id, a.video_path),
          snapshot:    a.snapshot_url ?? getSnapshotUrl(a.alert_id, a.snapshot_path),
        }));
      setSuspiciousActivities(prev => {
        const existingIds = new Set(prev.map(a => a.id));
        const newItems    = mapped.filter(a => !existingIds.has(a.id));
        return newItems.length > 0 ? [...newItems, ...prev] : prev;
      });
    } catch { }
  }, [resolvedAlertIds]);

  // ── Poll schedule ─────────────────────────────────────────────────────────
  useEffect(() => {
    fetchStats();
    fetchUnknown();
    setTimeout(() => fetchAlerts(), 300);
    setTimeout(() => fetchLogs(),   500);
  }, [fetchStats, fetchUnknown, fetchAlerts, fetchLogs]);

  useEffect(() => { const id = setInterval(fetchStats,    15_000); return () => clearInterval(id); }, [fetchStats]);
  useEffect(() => { const id = setInterval(fetchAlerts,   15_000); return () => clearInterval(id); }, [fetchAlerts]);
  useEffect(() => { const id = setInterval(fetchUnknown,   5_000); return () => clearInterval(id); }, [fetchUnknown]);
  useEffect(() => { const id = setInterval(fetchLogs,     10_000); return () => clearInterval(id); }, [fetchLogs]);

  const handleDailyAlertReset = async () => {
    if (!window.confirm('Archive all old active alerts for a fresh start today?')) return;
    setAlertResetLoading(true);
    try {
      const r = await fetch(`${API}/api/analytics/alerts/reset-daily`, { method: 'POST' });
      const d = await r.json();
      toast.success(`Archived ${d.archived} alerts — day reset complete`);
      fetchAlerts(); fetchStats();
    } catch { toast.error('Reset failed'); }
    finally { setAlertResetLoading(false); }
  };

  // ── WS detection handler ──────────────────────────────────────────────────
  const handleDetectionFromWS = (cameraId: string, detection: any) => {
    const key = `${cameraId}-${detection.timestamp}`;
    if (processedDetectionsRef.current.has(key)) return;
    processedDetectionsRef.current.add(key);

    // FIX-2: buffer if cameras not loaded yet
    if (!camerasLoadedRef.current) {
      earlyDetectionBuffer.current.push({ cameraId, detection });
      return;
    }

    // FIX-2: safe fallback — never crash if camera not found
    const camera = camerasRef.current.find(c => c.id === cameraId)
      ?? { id: cameraId, name: `Camera ${cameraId}`, location: '', status: 'active' as const };

    const cat: Person['category'] =
      detection.person?.classification === 'employee' ? 'Employee' :
      detection.person?.classification === 'owner'    ? 'Owner'    : 'Unknown';

    const newPerson: Person = {
      id:           key,
      name:         detection.person?.name || 'Unknown Person',
      category:     cat,
      confidence:   detection.confidence ?? 1,
      timestamp:    new Date(detection.timestamp),
      recognizedAs: detection.person?.name || undefined,
      imageData:    detection.face_image_url || undefined,
    };

    setPersons(prev => [newPerson, ...prev.slice(0, 199)]);
    setTimeLogs(prev => [{
      id:         `log-${key}`,
      personId:   newPerson.id,
      personName: newPerson.name!,
      category:   cat,
      inTime:     new Date(detection.timestamp),
      status:     'In' as const,
      cameraId,
      cameraName: camera.name,
      imageData:  newPerson.imageData,
    }, ...prev.slice(0, 499)]);

    // FIX-3: ONLY unknown persons go to the unknown list.
    // Registered employees/owners detected by the system MUST NOT appear
    // in the unknown notification popup — they are already classified.
    const isGenuinelyUnknown =
      cat === 'Unknown' &&
      !detection.person?.name &&
      !newPerson.recognizedAs &&
      !!detection.face_image_url;

    if (isGenuinelyUnknown) {
      const wsUnknown: Person = { ...newPerson, id: `ws-${key}` };
      setUnknownPersons(prev => {
        const alreadyHave = prev.some(u =>
          Math.abs(u.timestamp.getTime() - wsUnknown.timestamp.getTime()) < 5000 &&
          u.imageData === wsUnknown.imageData
        );
        if (alreadyHave) return prev;
        return [wsUnknown, ...prev.slice(0, 49)];
      });
    }
  };

  const handleDetection = (cameraId: string, detection: any) => {
    if (processedDetectionsRef.current.has(detection.id)) return;
    processedDetectionsRef.current.add(detection.id);
    const camera = camerasRef.current.find(c => c.id === cameraId);
    if (!camera) return;
    const cat: Person['category'] =
      detection.classification === 'employee' ? 'Employee' :
      detection.classification === 'owner'    ? 'Owner'    : 'Unknown';
    const newPerson: Person = {
      id: detection.id, name: detection.name || 'Unknown Person', category: cat,
      confidence: detection.confidence, timestamp: detection.timestamp,
      imageData: detection.face_image_url || undefined,
    };
    if (cat !== 'Unknown') toast.success(`✅ ${cat} recognized`, { description: newPerson.name });
    else toast.warning('❓ Unknown person detected!', { description: camera.name });
    setPersons(prev => [newPerson, ...prev]);
    const log: TimeLog = {
      id: `log-${detection.id}`, personId: newPerson.id, personName: newPerson.name!,
      category: cat, inTime: new Date(), status: 'In', cameraId, cameraName: camera.name,
      imageData: newPerson.imageData,
    };
    setTimeLogs(prev => [log, ...prev]);
  };

  // ── Classify — FIX-4: immediate removal from unknown list ─────────────────
  const handleClassify = async (
    id: string, name: string, category: 'Employee' | 'Owner' | 'Unknown',
  ) => {
    const numericId = parseInt(id, 10);
    if (!isNaN(numericId)) {
      try {
        const res = await fetch(`${API}/api/faces/unknown/${numericId}/classify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, classification: category.toLowerCase() }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast.error(`Classification failed: ${err.detail ?? `HTTP ${res.status}`}`);
          return;
        }
        const data = await res.json();
        const displayCategory: 'Employee' | 'Owner' | 'Unknown' =
          (data.category_display as any) ?? category;
        const merged: number = data.merged_duplicates ?? 0;
        toast.success(`✅ ${name} classified as ${displayCategory}`, {
          description: merged > 0
            ? `Face saved · ${merged} duplicate${merged !== 1 ? 's' : ''} auto-removed`
            : 'Face saved — will be auto-recognized next time',
          duration: 6000,
        });
        setPersons(prev => prev.map(p => p.id === id ? { ...p, name, category: displayCategory } : p));
        setTimeLogs(prev => prev.map(l => l.personId === id ? { ...l, personName: name, category: displayCategory } : l));
        // FIX-4: immediate removal, including ws- prefixed entries
        setUnknownPersons(prev => prev.filter(p =>
          p.id !== id && p.id !== `ws-${id}` && String(p.id) !== String(id)
        ));
      } catch {
        toast.error('Network error — classification not saved');
        return;
      }
    } else {
      toast.success(`${name} classified as ${category}`);
      setPersons(prev => prev.map(p => p.id === id ? { ...p, name, category } : p));
      setTimeLogs(prev => prev.map(l => l.personId === id ? { ...l, personName: name, category } : l));
      setUnknownPersons(prev => prev.filter(p => p.id !== id));
    }
    setClassifiedPersonIds(prev => new Set([...prev, id]));
    setTimeout(() => { fetchUnknown(); fetchStats(); fetchLogs(); }, 800);
  };

  const handleAlertResolved = useCallback((alertId: string) => {
    setResolvedAlertIds(prev => new Set([...prev, alertId]));
    setSuspiciousActivities(prev => prev.filter(a => a.id !== alertId));
  }, []);

  const handleAllAlertsResolved = useCallback(() => {
    setSuspiciousActivities(prev => {
      const ids = prev.map(a => a.id);
      setResolvedAlertIds(existing => new Set([...existing, ...ids]));
      return [];
    });
  }, []);

  const handleDownloadImage = async (person: Person) => {
    const url = person.imageData;
    if (!url) {
      const blob = new Blob(
        [`Person: ${person.name}\nCategory: ${person.category}\nTime: ${person.timestamp.toLocaleString()}`],
        { type: 'text/plain' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `person-${person.id}.txt`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      toast.success('Downloaded info file'); return;
    }
    try {
      const res = await fetch(url); if (!res.ok) throw new Error();
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `person_${person.id}.jpg`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href); toast.success('Image downloaded');
    } catch { window.open(url, '_blank'); }
  };

  const handleDownloadVideo = async (activity: SuspiciousActivity) => {
    const url = activity.videoClip?.startsWith('http') ? activity.videoClip : getAlertClipUrl(activity.id);
    try {
      const res = await fetch(url); if (!res.ok) throw new Error();
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `alert_${activity.id}.mp4`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href); toast.success('Video downloaded');
    } catch { window.open(url, '_blank'); toast.info('Opening video in new tab'); }
  };

  const handleExportLogs = () => {
    const csv = [
      ['Name', 'Category', 'Camera', 'In Time', 'Out Time', 'Status', 'Duration'],
      ...timeLogs.map(l => {
        const dur = l.outTime ? Math.round((l.outTime.getTime() - l.inTime.getTime()) / 60000) : 0;
        return [l.personName, l.category, l.cameraName,
          l.inTime.toLocaleString(), l.outTime?.toLocaleString() || 'N/A',
          l.status, `${dur} min`];
      }),
    ].map(r => r.join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `logs-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    toast.success('Logs exported');
  };

  const handleViewPerson = async (log: TimeLog) => {
    if (log.imageData) {
      setSelectedPerson({
        id: log.personId, name: log.personName,
        category: log.category as Person['category'],
        confidence: 1, timestamp: log.inTime, imageData: log.imageData,
      });
      setIsClassifierOpen(true); return;
    }
    const unknowns = unknownPersonsRef.current;
    const known    = personsRef.current;
    let person: Person | undefined =
      known.find(p => p.id === log.personId) ??
      unknowns.find(p => p.id === log.personId);
    if (!person && log.personName !== 'Unknown Person') {
      const byName = [...known, ...unknowns].find(p => p.name === log.personName && p.imageData);
      if (byName) person = { ...byName, name: log.personName, category: log.category as Person['category'] };
    }
    if (!person) {
      const logTime = log.inTime.getTime();
      const closest = unknowns.reduce<Person | null>((best, u) => {
        const diff = Math.abs(u.timestamp.getTime() - logTime);
        if (diff > 90_000) return best;
        if (!best) return u;
        return diff < Math.abs(best.timestamp.getTime() - logTime) ? u : best;
      }, null);
      if (closest) person = { ...closest, name: log.personName, category: log.category as Person['category'] };
    }
    if (!person && log.personId) {
      try {
        const r = await fetch(`${API}/api/faces/persons/${log.personId}`);
        if (r.ok) {
          const d = await r.json();
          person = { id: log.personId, name: d.name ?? log.personName,
            category: (d.classification === 'employee' ? 'Employee' : d.classification === 'owner' ? 'Owner' : 'Unknown') as Person['category'],
            confidence: 1, timestamp: log.inTime, imageData: d.face_image_url ?? undefined };
        }
      } catch { }
    }
    if (!person && log.personId) {
      try {
        const r = await fetch(`${API}/api/faces/unknown/${log.personId}`);
        if (r.ok) {
          const d = await r.json();
          person = { id: log.personId, name: log.personName, category: 'Unknown' as Person['category'],
            confidence: 1, timestamp: log.inTime, imageData: d.face_image_url ?? undefined };
        }
      } catch { }
    }
    setSelectedPerson(person ?? { id: log.personId, name: log.personName,
      category: log.category as Person['category'], confidence: 1, timestamp: log.inTime });
    setIsClassifierOpen(true);
  };

  const handleStatsCardClick = (type: 'total' | 'known' | 'unknown' | 'active' | 'alerts') => {
    if (type === 'unknown')     { setNotificationInitialTab('unknown'); setIsNotificationSliderOpen(true); }
    else if (type === 'alerts') { setNotificationInitialTab('alerts');  setIsNotificationSliderOpen(true); }
    else if (type === 'total')  { setNotificationInitialTab('recent');  setIsNotificationSliderOpen(true); }
    else if (type === 'known')  toast.info(`Known persons today: ${stats.known}`);
    else                        toast.info(`Currently on premises: ${stats.currentlyIn}`);
  };

  const handleRemoveCamera = async (cameraId: string) => {
    try { await fetch(`${API}/api/cameras/${cameraId}/stop`, { method: 'POST' }); } catch { }
    setCameras(prev => prev.filter(c => c.id !== cameraId));
    toast.success('Camera feed removed');
  };

  const handleDeleteCamera = (cameraId: string) => {
    setCameras(prev => prev.filter(c => c.id !== cameraId));
    toast.success('Camera deleted');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <Header username={username} role={role} notificationCount={stats.unknown + stats.alerts}
        onNotificationClick={() => { setNotificationInitialTab('alerts'); setIsNotificationSliderOpen(true); }}
        onLogout={onLogout} currentView={currentView} onViewChange={setCurrentView} />

      <div className="p-6 space-y-6">
        {currentView === 'home' && (
          <><StatsCards stats={stats} onCardClick={handleStatsCardClick} />
          <HomePage stats={stats} onDetection={handleDetection} /></>
        )}
        {currentView === 'about' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <AboutPage />
          </motion.div>
        )}

        {currentView === 'home' && (
          <Tabs defaultValue="cameras" className="space-y-4">
            <TabsList className="grid w-full grid-cols-4 h-14 bg-white shadow-md">
              <TabsTrigger value="cameras" className="gap-2 text-base data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white">
                <Camera className="w-5 h-5" /><span className="hidden sm:inline">📹 Cameras</span><span className="sm:hidden">📹</span>
              </TabsTrigger>
              <TabsTrigger value="add-camera" className="gap-2 text-base data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white">
                <Camera className="w-5 h-5" /><span className="hidden sm:inline">➕ Add Camera</span><span className="sm:hidden">➕</span>
              </TabsTrigger>
              <TabsTrigger value="logs" className="gap-2 text-base data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white">
                <Clock className="w-5 h-5" /><span className="hidden sm:inline">🕐 Time Logs</span><span className="sm:hidden">🕐</span>
              </TabsTrigger>
              <TabsTrigger value="alerts" className="gap-2 text-base data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white">
                <AlertTriangle className="w-5 h-5" /><span className="hidden sm:inline">⚠️ Alerts ({stats.alerts})</span><span className="sm:hidden">⚠️ {stats.alerts}</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="cameras" forceMount className="space-y-4 mt-0 data-[state=inactive]:hidden">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                {camerasLoading ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-4">
                    <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-gray-500 font-medium">Loading cameras…</p>
                  </div>
                ) : cameras.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
                    <div className="text-6xl">📷</div>
                    <p className="text-xl font-semibold text-gray-600">No cameras configured</p>
                    <p className="text-gray-500 max-w-md">Go to <strong>Add Camera</strong> tab.</p>
                  </div>
                ) : (
                  <CameraGrid cameras={cameras} onDetection={handleDetection}
                    onCameraDeleted={handleDeleteCamera} onCameraRemoved={handleRemoveCamera} />
                )}
              </motion.div>
            </TabsContent>

            <TabsContent value="add-camera" forceMount className="mt-0 data-[state=inactive]:hidden">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <AddCamera onAdded={(cam) => setCameras(prev => [...prev, cam])} />
              </motion.div>
            </TabsContent>

            <TabsContent value="logs" forceMount className="mt-0 data-[state=inactive]:hidden">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <TimeLogTable logs={timeLogs} onViewPerson={handleViewPerson}
                  onExport={handleExportLogs} onRefresh={() => { fetchLogs(); fetchUnknown(); }} />
              </motion.div>
            </TabsContent>

            <TabsContent value="alerts" forceMount className="mt-0 data-[state=inactive]:hidden">
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                <Card className="shadow-lg">
                  <CardHeader className="bg-gradient-to-r from-red-50 to-orange-50 border-b">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2 text-xl">⚠️ Security Threat Monitor</CardTitle>
                        <CardDescription className="text-base">Real-time suspicious activity detection and alerts</CardDescription>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="gap-2 text-xs border-green-300 text-green-700 hover:bg-green-50"
                          disabled={suspiciousActivities.filter(a => !resolvedAlertIds.has(a.id)).length === 0}
                          onClick={handleAllAlertsResolved}>✅ Resolve All</Button>
                        <Button variant="outline" size="sm" disabled={alertResetLoading} onClick={handleDailyAlertReset}
                          className="gap-2 text-xs border-orange-300 text-orange-700 hover:bg-orange-50">
                          <RefreshCw className={`w-3.5 h-3.5 ${alertResetLoading ? 'animate-spin' : ''}`} />Reset Day
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-3">
                      {suspiciousActivities.filter(a => !resolvedAlertIds.has(a.id)).length === 0 ? (
                        <div className="text-center py-12">
                          <div className="text-6xl mb-4">✅</div>
                          <div className="text-xl font-semibold text-gray-600">All Systems Secure</div>
                          <p className="text-gray-500 mt-2">No threats detected. Monitoring actively...</p>
                        </div>
                      ) : (
                        suspiciousActivities.filter(a => !resolvedAlertIds.has(a.id)).map((activity, index) => (
                          <motion.div key={activity.id}
                            initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: Math.min(index * 0.05, 0.3) }}
                            className="p-6 border-2 rounded-xl hover:shadow-xl transition-all cursor-pointer bg-gradient-to-r from-white to-red-50"
                            onClick={() => { setSelectedActivity(activity); setIsActivityAlertOpen(true); }}>
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-3">
                                <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                                  <AlertTriangle className="w-6 h-6 text-red-600" />
                                </div>
                                <div>
                                  <span className="text-xl font-bold capitalize">{activity.type.replace(/_/g,' ')}</span>
                                  <Badge variant="secondary" className={`ml-2 ${
                                    activity.severity === 'critical' ? 'bg-red-600 text-white'
                                    : activity.severity === 'high' ? 'bg-orange-600 text-white' : 'bg-yellow-600 text-white'
                                  }`}>{activity.severity}</Badge>
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <Button variant="outline" size="sm" className="text-xs text-green-700 border-green-200 hover:bg-green-50"
                                  onClick={(e) => { e.stopPropagation(); handleAlertResolved(activity.id); }}>✅ Resolve</Button>
                                <Button variant="outline" size="sm" className="gap-2"
                                  onClick={(e) => { e.stopPropagation(); setSelectedActivity(activity); setIsActivityAlertOpen(true); }}>
                                  <Video className="w-4 h-4" />View
                                </Button>
                              </div>
                            </div>
                            <p className="text-base text-gray-700 mb-3">{activity.description}</p>
                            <div className="flex items-center gap-4 text-sm text-gray-600">
                              <span>📹 {activity.cameraName}</span>
                              <span>•</span>
                              <span>🕐 {activity.timestamp.toLocaleString()}</span>
                              {activity.videoClip && <><span>•</span><span className="text-green-600 font-medium">🎬 Clip ready</span></>}
                            </div>
                          </motion.div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </TabsContent>
          </Tabs>
        )}

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          className="text-center py-4 bg-white rounded-lg shadow-md">
          <p className="text-sm text-gray-600">Powered by <span className="font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Neelaminds Private Limited</span></p>
          <p className="text-xs text-gray-500 mt-1">🇮🇳 Innovation • Security • Excellence</p>
        </motion.div>
      </div>

      {selectedPerson && (
        <PersonClassifier person={selectedPerson} isOpen={isClassifierOpen}
          onClose={() => setIsClassifierOpen(false)}
          onClassify={handleClassify} onDownload={handleDownloadImage} />
      )}

      {selectedActivity && (
        <SuspiciousActivityAlert activity={selectedActivity} isOpen={isActivityAlertOpen}
          onClose={() => { setIsActivityAlertOpen(false); setTimeout(() => setSelectedActivity(null), 300); }}
          onDownloadVideo={handleDownloadVideo} />
      )}

      <NotificationSlider
        isOpen={isNotificationSliderOpen}
        onClose={() => setIsNotificationSliderOpen(false)}
        unknownPersons={unknownPersons.filter(p => p.category === 'Unknown')}
        suspiciousActivities={suspiciousActivities}
        recentLogs={timeLogs}
        initialTab={notificationInitialTab}
        classifiedPersonIds={classifiedPersonIds}
        resolvedAlertIds={resolvedAlertIds}
        onClassifyPerson={(person) => { setSelectedPerson(person); setIsClassifierOpen(true); }}
        onDownloadImage={handleDownloadImage}
        onDownloadVideo={handleDownloadVideo}
        onAlertActioned={() => { fetchAlerts(); fetchStats(); }}
        onAlertResolved={handleAlertResolved}
        onAllAlertsResolved={handleAllAlertsResolved}
      />

      <MiniNotificationPopup
        unknownPersons={unknownPersons.filter(p => p.category === 'Unknown' && !p.recognizedAs)}
        suspiciousActivities={suspiciousActivities}
        resolvedAlertIds={resolvedAlertIds}
        onClassifyPerson={(person) => { setSelectedPerson(person); setIsClassifierOpen(true); }}
        onViewActivity={(activity) => { setSelectedActivity(activity); setIsActivityAlertOpen(true); }}
      />
    </div>
  );
}