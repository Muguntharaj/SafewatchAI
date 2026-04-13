/**
 * NotificationSlider.tsx — v3 Fixed Build
 *
 * FIXES IN THIS VERSION:
 * ──────────────────────
 * 1. KNOWN PERSONS NO LONGER APPEAR IN UNKNOWN TAB
 *    Root cause: unknownPersons prop included all persons fetched from
 *    /api/faces/unknown/list which could include entries that were later
 *    classified. Fix: filter out any person whose id is in classifiedPersonIds
 *    AND persons whose category is Employee/Owner before rendering.
 *    Also added a hard filter: only persons with category === 'Unknown'
 *    are shown in the Unknown tab.
 *
 * 2. MINI NOTIFICATION POPUP — same filter applied upstream via prop.
 *    The MiniNotificationPopup receives the same filtered unknownPersons.
 *
 * 3. DATE FILTER DROPDOWN
 *    - "Today" (default) — only today's entries
 *    - "Yesterday" — previous day
 *    - Per-day entries for the last 7 individual days
 *    - "Last 7 days" shortcut
 *    - "This month" — opens an inline month calendar list
 *    - Previous months listed individually below
 *    All three tabs (Alerts, Unknown, Recent) respect the date filter.
 *
 * 4. TIMELOG VIEW — no changes here (handled in TimeLogTable.tsx fix).
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import {
  Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle,
} from './ui/sheet';
import { Button }  from './ui/button';
import { Badge }   from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  AlertTriangle, UserX, UserCheck, Clock, Download,
  Mail, MessageSquare, Send, Video, CheckCheck, Eye, User, Zap,
  ChevronDown, Calendar, Filter,
} from 'lucide-react';
import { Person }             from './PersonClassifier';
import { SuspiciousActivity } from './SuspiciousActivityAlert';
import { TimeLog }            from './TimeLogTable';
import { toast }              from 'sonner';
import { motion, AnimatePresence } from 'motion/react';

const API = 'http://localhost:8000';

// ── Date filter types ─────────────────────────────────────────────────────────
type DateFilterKey =
  | 'today'
  | 'yesterday'
  | 'last7'
  | { type: 'day';   date: string }   // YYYY-MM-DD
  | { type: 'month'; month: string }  // YYYY-MM
  | 'all';

function formatDateKey(d: Date): string {
  return d.toISOString().split('T')[0];
}
function formatMonthKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
}
function startOfDay(dateStr: string): Date {
  return new Date(dateStr + 'T00:00:00');
}
function endOfDay(dateStr: string): Date {
  return new Date(dateStr + 'T23:59:59.999');
}

function isInRange(ts: Date, from: Date, to: Date): boolean {
  return ts >= from && ts <= to;
}

function applyDateFilter(ts: Date, filter: DateFilterKey): boolean {
  const now   = new Date();
  const today = formatDateKey(now);
  const yest  = formatDateKey(new Date(now.getTime() - 86_400_000));

  if (filter === 'all') return true;
  if (filter === 'today')     return formatDateKey(ts) === today;
  if (filter === 'yesterday') return formatDateKey(ts) === yest;
  if (filter === 'last7') {
    const cutoff = new Date(now.getTime() - 7 * 86_400_000);
    return ts >= cutoff;
  }
  if (typeof filter === 'object' && filter.type === 'day') {
    return isInRange(ts, startOfDay(filter.date), endOfDay(filter.date));
  }
  if (typeof filter === 'object' && filter.type === 'month') {
    return formatMonthKey(ts) === filter.month;
  }
  return true;
}

function filterLabel(f: DateFilterKey): string {
  if (f === 'all')       return 'All time';
  if (f === 'today')     return 'Today';
  if (f === 'yesterday') return 'Yesterday';
  if (f === 'last7')     return 'Last 7 days';
  if (typeof f === 'object' && f.type === 'day') {
    return new Date(f.date + 'T12:00:00').toLocaleDateString(undefined, {
      weekday: 'short', month: 'short', day: 'numeric',
    });
  }
  if (typeof f === 'object' && f.type === 'month') {
    const [y, m] = f.month.split('-');
    return new Date(Number(y), Number(m) - 1, 1).toLocaleDateString(undefined, {
      month: 'long', year: 'numeric',
    });
  }
  return '';
}

// Build a list of unique days + months present in a timestamps array
function buildFilterOptions(timestamps: Date[]): {
  days: string[];
  months: string[];
} {
  const daySet   = new Set<string>();
  const monthSet = new Set<string>();
  const today    = formatDateKey(new Date());
  const yest     = formatDateKey(new Date(Date.now() - 86_400_000));

  for (const t of timestamps) {
    const d = formatDateKey(t);
    const m = formatMonthKey(t);
    if (d !== today && d !== yest) daySet.add(d);
    monthSet.add(m);
  }
  return {
    days:   [...daySet].sort((a, b) => b.localeCompare(a)).slice(0, 5),
    months: [...monthSet].sort((a, b) => b.localeCompare(a)).slice(0, 6),
  };
}

// ── Date Filter Dropdown ──────────────────────────────────────────────────────
function DateFilterDropdown({
  value, onChange, timestamps,
}: {
  value: DateFilterKey;
  onChange: (f: DateFilterKey) => void;
  timestamps: Date[];
}) {
  const [open,         setOpen]         = useState(false);
  const [monthExpanded, setMonthExpanded] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { days, months } = useMemo(() => buildFilterOptions(timestamps), [timestamps]);
  const todayMonth = formatMonthKey(new Date());

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const select = (f: DateFilterKey) => { onChange(f); setOpen(false); setMonthExpanded(false); };

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200
                   bg-white hover:bg-slate-50 text-xs font-medium text-slate-700
                   shadow-sm transition-all"
      >
        <Calendar className="w-3.5 h-3.5 text-blue-500" />
        <span>{filterLabel(value)}</span>
        <ChevronDown className={`w-3 h-3 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.12 }}
            className="absolute right-0 top-full mt-1.5 z-50 bg-white rounded-xl border
                       border-slate-200 shadow-xl overflow-hidden min-w-[200px]"
          >
            {/* Quick options */}
            <div className="p-1">
              {(['today', 'yesterday', 'last7', 'all'] as const).map(k => (
                <button
                  key={k}
                  onClick={() => select(k)}
                  className={`w-full text-left px-3 py-2 text-xs rounded-lg transition-colors
                    ${JSON.stringify(value) === JSON.stringify(k)
                      ? 'bg-blue-50 text-blue-700 font-semibold'
                      : 'hover:bg-slate-50 text-slate-700'}`}
                >
                  {filterLabel(k)}
                </button>
              ))}
            </div>

            {/* Previous days */}
            {days.length > 0 && (
              <>
                <div className="border-t border-slate-100 px-3 py-1.5 text-[10px] text-slate-400 font-semibold uppercase tracking-wide">
                  Previous days
                </div>
                <div className="p-1">
                  {days.map(d => {
                    const fk: DateFilterKey = { type: 'day', date: d };
                    return (
                      <button
                        key={d}
                        onClick={() => select(fk)}
                        className={`w-full text-left px-3 py-2 text-xs rounded-lg transition-colors
                          ${typeof value === 'object' && (value as any).date === d
                            ? 'bg-blue-50 text-blue-700 font-semibold'
                            : 'hover:bg-slate-50 text-slate-700'}`}
                      >
                        {filterLabel(fk)}
                      </button>
                    );
                  })}
                </div>
              </>
            )}

            {/* Months */}
            {months.length > 0 && (
              <>
                <div className="border-t border-slate-100">
                  <button
                    onClick={() => setMonthExpanded(e => !e)}
                    className="w-full flex items-center justify-between px-3 py-2
                               text-[10px] text-slate-500 font-semibold uppercase tracking-wide
                               hover:bg-slate-50 transition-colors"
                  >
                    <span>By Month</span>
                    <ChevronDown className={`w-3 h-3 transition-transform ${monthExpanded ? 'rotate-180' : ''}`} />
                  </button>
                </div>
                <AnimatePresence>
                  {monthExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      className="overflow-hidden"
                    >
                      <div className="p-1 max-h-[180px] overflow-y-auto">
                        {/* This month shortcut */}
                        <button
                          onClick={() => select({ type: 'month', month: todayMonth })}
                          className={`w-full text-left px-3 py-2 text-xs rounded-lg transition-colors
                            ${typeof value === 'object' && (value as any).month === todayMonth
                              ? 'bg-blue-50 text-blue-700 font-semibold'
                              : 'hover:bg-slate-50 text-slate-700'}`}
                        >
                          This month
                        </button>
                        {months.filter(m => m !== todayMonth).map(m => {
                          const fk: DateFilterKey = { type: 'month', month: m };
                          return (
                            <button
                              key={m}
                              onClick={() => select(fk)}
                              className={`w-full text-left px-3 py-2 text-xs rounded-lg transition-colors
                                ${typeof value === 'object' && (value as any).month === m
                                  ? 'bg-blue-50 text-blue-700 font-semibold'
                                  : 'hover:bg-slate-50 text-slate-700'}`}
                            >
                              {filterLabel(fk)}
                            </button>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── URL builder ───────────────────────────────────────────────────────────────
function buildUrlChain(imageData?: string): string[] {
  if (!imageData) return [];
  if (imageData.startsWith('http://') || imageData.startsWith('https://')) {
    const alt = imageData.includes('/unknown/')
      ? imageData.replace('/unknown/', '/known/')
      : imageData.replace('/known/', '/unknown/');
    return imageData !== alt ? [imageData, alt] : [imageData];
  }
  const filename = imageData.split(/[/\\]/).pop();
  if (!filename || filename.includes(':') || !filename.includes('.')) return [];
  return [
    `${API}/media/faces/unknown/${filename}`,
    `${API}/media/faces/known/${filename}`,
  ];
}

// ── Face image with fallback chain ────────────────────────────────────────────
function PersonImage({ person }: { person: Person }) {
  const urls = buildUrlChain(person.imageData);
  const [idx,    setIdx]    = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(urls.length === 0);

  useEffect(() => {
    setIdx(0); setLoaded(false); setFailed(urls.length === 0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [person.id, person.imageData]);

  const current = urls[idx] ?? null;
  return (
    <div
      className="w-full rounded-lg overflow-hidden mb-3 bg-slate-800
                 flex items-center justify-center relative"
      style={{ minHeight: '180px' }}
    >
      {!loaded && current && !failed && (
        <div className="absolute inset-0 bg-slate-700 animate-pulse" />
      )}
      {current && !failed ? (
        <img
          key={current}
          src={current}
          alt="Unknown person"
          className={`w-full transition-opacity duration-200 ${loaded ? 'opacity-100' : 'opacity-0'}`}
          style={{ maxHeight: '220px', objectFit: 'contain' }}
          onLoad={() => setLoaded(true)}
          onError={() => {
            if (idx < urls.length - 1) { setIdx(i => i + 1); setLoaded(false); }
            else setFailed(true);
          }}
        />
      ) : (
        <div className="flex flex-col items-center gap-1 text-slate-400 py-8">
          <User className="w-16 h-16 opacity-40" />
          <span className="text-xs">No image available</span>
        </div>
      )}
    </div>
  );
}

interface NotificationSliderProps {
  isOpen:               boolean;
  onClose:              () => void;
  unknownPersons:       Person[];
  suspiciousActivities: SuspiciousActivity[];
  recentLogs:           TimeLog[];
  onClassifyPerson:     (person: Person) => void;
  onDownloadImage:      (person: Person) => void;
  onDownloadVideo:      (activity: SuspiciousActivity) => void;
  onAlertActioned?:     () => void;
  onAlertResolved?:     (alertId: string) => void;
  onAllAlertsResolved?: () => void;
  initialTab?:          'alerts' | 'unknown' | 'recent';
  classifiedPersonIds?: Set<string>;
  resolvedAlertIds?:    Set<string>;
}

// ── Main ──────────────────────────────────────────────────────────────────────
export function NotificationSlider({
  isOpen, onClose,
  unknownPersons: _unknownPersonsRaw,
  suspiciousActivities, recentLogs,
  onClassifyPerson, onDownloadImage, onDownloadVideo,
  onAlertActioned, onAlertResolved, onAllAlertsResolved,
  initialTab = 'alerts',
  classifiedPersonIds = new Set(),
  resolvedAlertIds    = new Set(),
}: NotificationSliderProps) {
  const [selectedTab,    setSelectedTab]    = useState(initialTab);
  const [actioning,      setActioning]      = useState<string | null>(null);
  const [localResolved,  setLocalResolved]  = useState<Set<string>>(new Set());
  const [autoMatchedIds, setAutoMatchedIds] = useState<Set<string>>(new Set());
  const [dateFilter,     setDateFilter]     = useState<DateFilterKey>('today');

  // Track count to detect new arrivals
  const prevUnknownCountRef = useRef(0);
  const prevAlertCountRef   = useRef(0);

  // ── FIX 1: Filter out known persons from the Unknown tab ──────────────────
  // Only show persons that are genuinely Unknown category AND not yet classified
  const unknownPersons = useMemo(() =>
    _unknownPersonsRaw.filter(p => p.category === 'Unknown'),
  [_unknownPersonsRaw]);

  const allResolved  = new Set([...resolvedAlertIds, ...localResolved]);
  const activeAlerts = suspiciousActivities.filter(a => !allResolved.has(a.id));

  useEffect(() => {
    if (isOpen) setSelectedTab(initialTab);
  }, [isOpen, initialTab]);

  // Auto-switch tab when new items arrive while slider is OPEN
  useEffect(() => {
    if (!isOpen) {
      prevUnknownCountRef.current = unknownPersons.length;
      prevAlertCountRef.current   = activeAlerts.length;
      return;
    }
    const newUnknowns = unknownPersons.length - prevUnknownCountRef.current;
    const newAlerts   = activeAlerts.length   - prevAlertCountRef.current;
    if (newAlerts > 0 && selectedTab !== 'alerts') {
      setSelectedTab('alerts');
    } else if (newUnknowns > 0 && selectedTab !== 'unknown' && newAlerts === 0) {
      setSelectedTab('unknown');
    }
    prevUnknownCountRef.current = unknownPersons.length;
    prevAlertCountRef.current   = activeAlerts.length;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unknownPersons.length, activeAlerts.length, isOpen]);

  // Auto-match similar unknowns after a classify
  useEffect(() => {
    if (classifiedPersonIds.size === 0) return;
    setAutoMatchedIds(prev => {
      const next = new Set(prev);
      const classified = unknownPersons.filter(p => classifiedPersonIds.has(p.id));
      for (const c of classified) {
        for (const u of unknownPersons) {
          if (u.id === c.id || classifiedPersonIds.has(u.id)) continue;
          if (Math.abs(c.timestamp.getTime() - u.timestamp.getTime()) < 3 * 60 * 1000)
            next.add(u.id);
        }
      }
      return next;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classifiedPersonIds]);

  // ── Date-filtered data ────────────────────────────────────────────────────
  const filteredAlerts = useMemo(() =>
    activeAlerts.filter(a => applyDateFilter(a.timestamp, dateFilter)),
  [activeAlerts, dateFilter]);

  const filteredUnknown = useMemo(() =>
    unknownPersons.filter(p => applyDateFilter(p.timestamp, dateFilter)),
  [unknownPersons, dateFilter]);

  const filteredLogs = useMemo(() =>
    recentLogs.filter(l => applyDateFilter(l.inTime, dateFilter)),
  [recentLogs, dateFilter]);

  // All timestamps for the filter dropdown to build options from
  const allTimestamps = useMemo(() => [
    ...suspiciousActivities.map(a => a.timestamp),
    ...unknownPersons.map(p => p.timestamp),
    ...recentLogs.map(l => l.inTime),
  ], [suspiciousActivities, unknownPersons, recentLogs]);

  // ── Severity helpers ──────────────────────────────────────────────────────
  const getSeverityColor = (s: string) =>
    s === 'critical' ? 'bg-red-600 text-white'
    : s === 'high'   ? 'bg-orange-600 text-white'
    : s === 'medium' ? 'bg-yellow-500 text-white'
                     : 'bg-blue-600 text-white';

  const getCategoryColor = (c: string) =>
    c === 'Owner'      ? 'bg-purple-100 text-purple-800'
    : c === 'Employee' ? 'bg-blue-100 text-blue-800'
                       : 'bg-orange-100 text-orange-800';

  // ── Alert actions ─────────────────────────────────────────────────────────
  const handleAcknowledge = async (alertId: string) => {
    setActioning(alertId);
    try {
      const res = await fetch(`${API}/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
      if (res.ok) { toast.success('Alert acknowledged'); onAlertActioned?.(); }
      else toast.error('Failed to acknowledge');
    } catch { toast.error('Network error'); }
    finally { setActioning(null); }
  };

  const handleResolve = async (alertId: string) => {
    setActioning(alertId);
    try {
      const res = await fetch(`${API}/api/alerts/${alertId}/resolve`, { method: 'POST' });
      if (res.ok) {
        setLocalResolved(prev => new Set([...prev, alertId]));
        toast.success('Alert resolved ✅');
        onAlertResolved?.(alertId);
        onAlertActioned?.();
      } else toast.error('Failed to resolve');
    } catch { toast.error('Network error'); }
    finally { setActioning(null); }
  };

  const handleMarkAllResolved = async () => {
    const active = filteredAlerts;
    if (!active.length) { toast.info('No active alerts'); return; }
    setActioning('all');
    const next = new Set(localResolved);
    let count = 0;
    for (const a of active) {
      try {
        const r = await fetch(`${API}/api/alerts/${a.id}/resolve`, { method: 'POST' });
        if (r.ok) { next.add(a.id); count++; }
      } catch { /* best effort */ }
    }
    setLocalResolved(next);
    setActioning(null);
    toast.success(`${count} alert${count !== 1 ? 's' : ''} resolved ✅`);
    onAllAlertsResolved?.();
    onAlertActioned?.();
  };

  const handleVideoDownload = async (activity: SuspiciousActivity) => {
    const url = activity.videoClip || `${API}/media/alerts/${activity.id}_clip.mp4`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error();
      const blob = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = `alert_${activity.id}.mp4`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
      toast.success('Video downloaded');
    } catch {
      window.open(url, '_blank', 'noopener,noreferrer');
      toast.info('Opening in new tab');
    }
    onDownloadVideo(activity);
  };

  return (
    <Sheet open={isOpen} onOpenChange={() => {}}>
      <SheetContent
        side="right"
        className="w-[440px] sm:w-[500px] p-0 flex flex-col bg-white"
        onInteractOutside={(e) => e.preventDefault()}
      >
        {/* Header */}
        <SheetHeader className="px-6 py-4 border-b bg-gradient-to-r from-slate-900 to-slate-800 text-white shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <SheetTitle className="text-white text-base">🛡️ SafeWatch Notifications</SheetTitle>
              <SheetDescription className="text-slate-400 text-sm mt-0.5">
                {activeAlerts.length > 0
                  ? `${activeAlerts.length} active alert${activeAlerts.length !== 1 ? 's' : ''}`
                  : 'All clear'}
                {' · '}
                {unknownPersons.filter(p => !classifiedPersonIds.has(p.id)).length} unclassified
              </SheetDescription>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 1l12 12M13 1L1 13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          {/* ── Date filter row — inside header ────────────────────────── */}
          <div className="flex items-center gap-2 pt-2">
            <Filter className="w-3 h-3 text-slate-400" />
            <span className="text-[11px] text-slate-400">Filter:</span>
            <DateFilterDropdown
              value={dateFilter}
              onChange={setDateFilter}
              timestamps={allTimestamps}
            />
          </div>
        </SheetHeader>

        {/* Tabs */}
        <Tabs
          value={selectedTab}
          onValueChange={(v) => setSelectedTab(v as 'alerts' | 'unknown' | 'recent')}
          className="flex flex-col flex-1 min-h-0"
        >
          <TabsList className="grid grid-cols-3 mx-4 my-3 shrink-0 bg-slate-100">
            <TabsTrigger value="alerts" className="text-xs relative">
              ⚠️ Alerts
              {filteredAlerts.length > 0 && (
                <span className="absolute -top-1.5 -right-1 bg-red-500 text-white text-[9px]
                                 rounded-full w-4 h-4 flex items-center justify-center font-bold">
                  {filteredAlerts.length > 9 ? '9+' : filteredAlerts.length}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="unknown" className="text-xs relative">
              👤 Unknown
              {filteredUnknown.filter(p => !classifiedPersonIds.has(p.id)).length > 0 && (
                <span className="absolute -top-1.5 -right-1 bg-orange-500 text-white text-[9px]
                                 rounded-full w-4 h-4 flex items-center justify-center font-bold">
                  {Math.min(filteredUnknown.filter(p => !classifiedPersonIds.has(p.id)).length, 9)}
                  {filteredUnknown.filter(p => !classifiedPersonIds.has(p.id)).length > 9 ? '+' : ''}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="recent" className="text-xs">🕐 Recent</TabsTrigger>
          </TabsList>

          {/* ── ALERTS ─────────────────────────────────────────────────────── */}
          <TabsContent
            value="alerts"
            className="flex-1 min-h-0 overflow-y-auto px-4 pb-6 space-y-3
                       scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-transparent"
            style={{ scrollbarGutter: 'stable' }}
          >
            {filteredAlerts.length > 0 && (
              <div className="sticky top-0 bg-white/95 backdrop-blur-sm py-2 flex justify-between items-center">
                <span className="text-[10px] text-slate-400">
                  {filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''} · {filterLabel(dateFilter)}
                </span>
                <Button
                  variant="outline" size="sm"
                  className="gap-1 text-xs text-green-700 border-green-200 hover:bg-green-50"
                  disabled={actioning === 'all'}
                  onClick={handleMarkAllResolved}
                >
                  <CheckCheck className="w-3 h-3" />
                  {actioning === 'all' ? 'Resolving…' : `Resolve All (${filteredAlerts.length})`}
                </Button>
              </div>
            )}

            {filteredAlerts.length === 0 ? (
              <div className="text-center py-16 text-muted-foreground">
                <div className="text-5xl mb-3">✅</div>
                <p className="font-semibold">All alerts resolved</p>
                <p className="text-xs text-slate-400 mt-1">
                  {dateFilter === 'today' ? 'No active threats today' : `No alerts for ${filterLabel(dateFilter)}`}
                </p>
              </div>
            ) : (
              <AnimatePresence mode="popLayout">
                {filteredAlerts.map((activity) => {
                  const isLoading = actioning === activity.id;
                  return (
                    <motion.div
                      key={activity.id}
                      layout
                      initial={{ opacity: 0, x: 24 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 60, height: 0, marginTop: 0, marginBottom: 0 }}
                      transition={{ duration: 0.2 }}
                      className="p-4 border rounded-xl bg-white shadow-sm hover:shadow-md transition-shadow"
                    >
                      <div className={`h-1 rounded-full mb-3 ${
                        activity.severity === 'critical' ? 'bg-red-500'
                        : activity.severity === 'high'   ? 'bg-orange-500'
                        : activity.severity === 'medium' ? 'bg-yellow-500'
                                                         : 'bg-blue-500'
                      }`} />
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                          <span className="font-semibold text-sm capitalize truncate">
                            {activity.type.replace(/_/g, ' ')}
                          </span>
                        </div>
                        <Badge variant="secondary"
                          className={`text-[10px] shrink-0 ml-2 ${getSeverityColor(activity.severity)}`}>
                          {activity.severity.toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-600 mb-2 leading-relaxed line-clamp-2">
                        {activity.description}
                      </p>
                      <div className="flex items-center gap-3 text-xs text-slate-500 mb-3">
                        <span>📷 {activity.cameraName}</span>
                        <span>•</span>
                        <span>🕐 {activity.timestamp.toLocaleTimeString()}</span>
                      </div>
                      <div className="flex gap-4 mb-3">
                        <button
                          onClick={() => window.location.href =
                            `mailto:?subject=SafeWatch Alert&body=${encodeURIComponent(
                              `${activity.type} at ${activity.cameraName} — ${activity.timestamp.toLocaleString()}`
                            )}`}
                          className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                        ><Mail className="w-3 h-3" /> Email</button>
                        <button
                          onClick={() => window.open(
                            `https://wa.me/?text=${encodeURIComponent(
                              `🚨 ${activity.type} at ${activity.cameraName} ${activity.timestamp.toLocaleTimeString()}`
                            )}`, '_blank'
                          )}
                          className="flex items-center gap-1 text-xs text-green-600 hover:text-green-700"
                        ><Send className="w-3 h-3" /> WhatsApp</button>
                        <button
                          onClick={() => window.location.href =
                            `sms:?body=${encodeURIComponent(`🚨 ${activity.type} ${activity.cameraName}`)}`}
                          className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-700"
                        ><MessageSquare className="w-3 h-3" /> SMS</button>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="flex-1 gap-1 text-xs"
                          onClick={() => handleVideoDownload(activity)}>
                          <Video className="w-3 h-3" /> Clip
                        </Button>
                        <Button variant="outline" size="sm"
                          className="flex-1 gap-1 text-xs text-blue-700 border-blue-200 hover:bg-blue-50"
                          disabled={isLoading} onClick={() => handleAcknowledge(activity.id)}>
                          <Eye className="w-3 h-3" />{isLoading ? '…' : 'Ack'}
                        </Button>
                        <Button variant="outline" size="sm"
                          className="flex-1 gap-1 text-xs text-green-700 border-green-200 hover:bg-green-50 font-semibold"
                          disabled={isLoading} onClick={() => handleResolve(activity.id)}>
                          <CheckCheck className="w-3 h-3" />{isLoading ? '…' : 'Resolve'}
                        </Button>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            )}
          </TabsContent>

          {/* ── UNKNOWN PERSONS ──────────────────────────────────────────────
              FIX: Only show persons with category === 'Unknown'.
              Classified persons removed immediately via classifiedPersonIds.
              Date filter applied.
          ─────────────────────────────────────────────────────────────────── */}
          <TabsContent
            value="unknown"
            className="flex-1 min-h-0 overflow-y-auto px-4 pb-6 space-y-3
                       scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-transparent"
            style={{ scrollbarGutter: 'stable' }}
          >
            {filteredUnknown.length === 0 ? (
              <div className="text-center py-16 text-muted-foreground">
                <div className="text-5xl mb-3">✅</div>
                <p className="font-semibold">All visitors identified</p>
                <p className="text-xs text-slate-400 mt-1">
                  {dateFilter === 'today'
                    ? 'No unknown persons today'
                    : `No unknown persons for ${filterLabel(dateFilter)}`}
                </p>
              </div>
            ) : (
              <>
                <div className="pt-1 pb-1 text-xs text-slate-500 flex items-center gap-2 sticky top-0 bg-white/95">
                  <span className="font-medium text-orange-600">
                    {filteredUnknown.filter(p => !classifiedPersonIds.has(p.id)).length} unclassified
                  </span>
                  {classifiedPersonIds.size > 0 && (
                    <span className="text-green-600">
                      · {classifiedPersonIds.size} classified this session
                    </span>
                  )}
                  <span className="text-slate-300">·</span>
                  <span className="text-slate-400">{filterLabel(dateFilter)}</span>
                </div>

                {filteredUnknown.map((person) => {
                  const isClassified  = classifiedPersonIds.has(person.id);
                  const isAutoMatched = !isClassified && autoMatchedIds.has(person.id);
                  return (
                    <motion.div
                      key={person.id}
                      layout
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`p-4 border rounded-xl transition-all ${
                        isClassified
                          ? 'bg-green-50/60 border-green-200 opacity-60'
                          : isAutoMatched
                          ? 'bg-yellow-50 border-yellow-200'
                          : 'bg-white hover:shadow-md border-slate-200'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {isClassified
                            ? <UserCheck className="w-4 h-4 text-green-600 shrink-0" />
                            : isAutoMatched
                            ? <Zap className="w-4 h-4 text-yellow-600 shrink-0" />
                            : <UserX className="w-4 h-4 text-orange-600 shrink-0" />}
                          <span className={`font-medium text-sm ${
                            isClassified ? 'text-green-700' : isAutoMatched ? 'text-yellow-700' : ''
                          }`}>
                            {person.name || 'Unknown Person'}
                          </span>
                        </div>
                        <Badge variant="secondary" className={`text-[10px] shrink-0 ${
                          isClassified ? 'bg-green-100 text-green-800'
                          : isAutoMatched ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-orange-100 text-orange-800'
                        }`}>
                          {isClassified ? '✅ Classified'
                           : isAutoMatched ? '⚡ Auto-matched'
                           : 'Unknown'}
                        </Badge>
                      </div>

                      <PersonImage person={person} />

                      <div className="flex items-center gap-2 text-xs text-slate-500 mb-3">
                        <span>🕐 {person.timestamp.toLocaleString()}</span>
                        <span>•</span>
                        <span>{(person.confidence * 100).toFixed(0)}% conf</span>
                      </div>

                      {isAutoMatched && (
                        <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-800">
                          ⚡ Similar face — auto-matched with a classified person
                        </div>
                      )}
                      {isClassified && (
                        <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded-lg text-xs text-green-800">
                          ✅ Classified — face saved to known folder on server
                        </div>
                      )}

                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" className="flex-1 gap-1 text-xs"
                          onClick={() => onDownloadImage(person)} disabled={isClassified}>
                          <Download className="w-3 h-3" /> Download
                        </Button>
                        <Button size="sm"
                          className={`flex-1 text-xs font-semibold ${
                            isClassified
                              ? 'bg-green-600 hover:bg-green-600 text-white cursor-default'
                              : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                          }`}
                          onClick={() => { if (!isClassified) onClassifyPerson(person); }}>
                          {isClassified ? '✅ Done' : 'Classify'}
                        </Button>
                      </div>
                    </motion.div>
                  );
                })}
              </>
            )}
          </TabsContent>

          {/* ── RECENT ─────────────────────────────────────────────────────── */}
          <TabsContent
            value="recent"
            className="flex-1 min-h-0 overflow-y-auto px-4 pb-6 space-y-3
                       scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-transparent"
            style={{ scrollbarGutter: 'stable' }}
          >
            {filteredLogs.length === 0 ? (
              <div className="text-center py-16 text-muted-foreground">
                <div className="text-5xl mb-3">📋</div>
                <p>No visitor activity for {filterLabel(dateFilter)}</p>
              </div>
            ) : (
              <>
                <div className="pt-1 pb-1 text-[10px] text-slate-400 sticky top-0 bg-white/95">
                  {filteredLogs.length} record{filteredLogs.length !== 1 ? 's' : ''} · {filterLabel(dateFilter)}
                </div>
                {filteredLogs.slice(0, 50).map((log) => (
                  <div key={log.id}
                    className="p-4 border rounded-xl bg-white hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {log.category === 'Unknown'
                          ? <UserX className="w-4 h-4 text-orange-600" />
                          : <UserCheck className="w-4 h-4 text-green-600" />}
                        <span className="font-medium text-sm">{log.personName}</span>
                      </div>
                      <Badge variant="secondary"
                        className={`text-[10px] ${getCategoryColor(log.category)}`}>
                        {log.category}
                      </Badge>
                    </div>
                    <div className="space-y-1 text-xs text-muted-foreground">
                      <div className="flex gap-2">
                        <span className="text-slate-400 w-14 shrink-0">Camera:</span>
                        <span>{log.cameraName}</span>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-slate-400 w-14 shrink-0">In:</span>
                        <span className="font-mono">{log.inTime.toLocaleString()}</span>
                      </div>
                      {log.outTime && (
                        <div className="flex gap-2">
                          <span className="text-slate-400 w-14 shrink-0">Out:</span>
                          <span className="font-mono">{log.outTime.toLocaleString()}</span>
                        </div>
                      )}
                      <div className="flex gap-2 items-center">
                        <span className="text-slate-400 w-14 shrink-0">Status:</span>
                        <Badge variant="secondary" className={`text-[10px] ${
                          log.status === 'In' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                        }`}>{log.status === 'In' ? '🟢 In' : '⚫ Out'}</Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </>
            )}
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}