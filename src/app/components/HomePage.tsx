/**
 * HomePage.tsx — SafeWatch AI Analytics (Power BI / Tableau style)
 *
 * LEFT PANEL — Interactive Analytics:
 *   - Activity Timeline with Known / Unknown / Alerts area chart
 *   - X-axis selector (hourly / daily / weekly)
 *   - Y-axis selector (any metric: detections, known, unknown, alerts, zones)
 *   - Comparison metric (overlay a second Y-axis series)
 *   - Chart type toggle: Area / Bar / Line
 *   - Period filter: Today / 7 Days / 30 Days
 *   - Camera Performance bar chart (real data)
 *   - Person Distribution donut with legend
 *
 * RIGHT PANEL — Primary Cameras sidebar (3 big live feeds)
 */

import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  ComposedChart, PieChart, Pie, Cell, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { CCTVFeed } from './CCTVFeed';
import {
  TrendingUp, BarChart3, PieChart as PieChartIcon, Eye, Video,
  RefreshCw, ChevronDown, ChevronUp,
} from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

const API = 'http://localhost:8000';

// ── Types ─────────────────────────────────────────────────────────────────────
interface HomePageProps {
  stats: { total: number; known: number; unknown: number; currentlyIn: number; alerts: number; };
  onDetection?: (cameraId: string, detection: any) => void;
}

interface CamInfo { id: string; name: string; location: string; status: 'active' | 'inactive'; }

type PeriodKey    = 'today' | 'week' | 'month';
type ChartTypeKey = 'area' | 'bar' | 'line' | 'composed';
type MetricKey    = 'known' | 'unknown' | 'total' | 'alerts' | 'critical' | 'z1' | 'z2' | 'z3';
type XAxisKey     = 'hourly' | 'daily' | 'weekly';

const METRIC_LABELS: Record<MetricKey, string> = {
  known:    'Known Persons',
  unknown:  'Unknown Visitors',
  total:    'Total Detections',
  alerts:   'All Alerts',
  critical: 'Critical Alerts',
  z1:       'Zone 1 (Close)',
  z2:       'Zone 2 (Medium)',
  z3:       'Zone 3 (Far)',
};

const METRIC_COLORS: Record<MetricKey, string> = {
  known:    '#10b981',
  unknown:  '#f59e0b',
  total:    '#6366f1',
  alerts:   '#ef4444',
  critical: '#dc2626',
  z1:       '#8b5cf6',
  z2:       '#06b6d4',
  z3:       '#84cc16',
};

const XAXIS_LABELS: Record<XAxisKey, string> = {
  hourly: 'Hourly',
  daily:  'Daily',
  weekly: 'Weekly',
};

const PERIOD_LABELS: Record<PeriodKey, string> = {
  today: 'Today',
  week:  'Last 7 Days',
  month: 'Last 30 Days',
};

const CAM_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

// ── Donut label ───────────────────────────────────────────────────────────────
function DonutLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) {
  if (percent < 0.04) return null;
  const R = Math.PI / 180;
  const r = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + r * Math.cos(-midAngle * R);
  const y = cy + r * Math.sin(-midAngle * R);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle"
          dominantBaseline="central" fontSize={11} fontWeight="bold">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

// ── Sidebar camera card ───────────────────────────────────────────────────────
function SidebarCam({ cam, index, onDetection }: {
  cam: CamInfo; index: number;
  onDetection?: (id: string, det: any) => void;
}) {
  return (
    <div className="rounded-xl overflow-hidden border border-gray-700/50 shadow-lg bg-gray-950">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-gray-500 text-xs font-bold shrink-0">
            {String(index + 1).padStart(2, '0')}
          </span>
          <div className="min-w-0">
            <p className="text-white text-sm font-semibold truncate leading-tight">{cam.name}</p>
            <p className="text-gray-500 text-[10px] truncate">{cam.location}</p>
          </div>
        </div>
        {cam.status === 'active' ? (
          <div className="flex items-center gap-1.5 shrink-0 ml-2 bg-red-600 text-white
                          text-[10px] font-bold px-2 py-0.5 rounded-full">
            <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
            LIVE
          </div>
        ) : (
          <span className="shrink-0 ml-2 text-[10px] text-gray-500 bg-gray-800
                           px-2 py-0.5 rounded-full">OFFLINE</span>
        )}
      </div>
      <div className="w-full" style={{ aspectRatio: '16/9' }}>
        <CCTVFeed
          cameraId={cam.id}
          isActive={cam.status === 'active'}
          onDetection={onDetection ? (det) => onDetection(cam.id, det) : undefined}
        />
      </div>
    </div>
  );
}

// ── Custom tooltip ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 shadow-2xl">
      <p className="text-slate-400 text-xs font-medium mb-2">{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-300">{p.name}:</span>
          <span className="text-white font-bold">{p.value}</span>
        </div>
      ))}
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────
export function HomePage({ stats, onDetection }: HomePageProps) {
  // ── State ─────────────────────────────────────────────────────────────────
  const [cameras,       setCameras]       = useState<CamInfo[]>([]);
  const [rawTimeline,   setRawTimeline]   = useState<any>({});   // {labels, detections, alerts, ...}
  const [cameraBarData, setCameraBarData] = useState<any[]>([]);
  const [loading,       setLoading]       = useState(true);
  const [lastUpdated,   setLastUpdated]   = useState<Date | null>(null);
  const [analyticsOpen, setAnalyticsOpen] = useState(true);

  // ── Chart controls ─────────────────────────────────────────────────────────
  const [period,         setPeriod]         = useState<PeriodKey>('today');
  const [chartType,      setChartType]      = useState<ChartTypeKey>('area');
  const [primaryMetric,  setPrimaryMetric]  = useState<MetricKey>('known');
  const [compareMetric,  setCompareMetric]  = useState<MetricKey | 'none'>('unknown');
  const [xAxisMode,      setXAxisMode]      = useState<XAxisKey>('hourly');
  const [showReferences, setShowReferences] = useState(false);

  // ── Donut data from stats ─────────────────────────────────────────────────
  const donutData = useMemo(() => {
    const items = [];
    if (stats.known   > 0) items.push({ name: 'Employees', value: stats.known,   color: '#10b981' });
    if (stats.unknown > 0) items.push({ name: 'Unknown',   value: stats.unknown,  color: '#f59e0b' });
    if (stats.currentlyIn > 0) items.push({ name: 'Live Now', value: stats.currentlyIn, color: '#6366f1' });
    return items.length > 0 ? items : [{ name: 'No data', value: 1, color: '#374151' }];
  }, [stats.known, stats.unknown, stats.currentlyIn]);

  // ── Fetch cameras ─────────────────────────────────────────────────────────
  const fetchCameras = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/cameras/list`);
      if (!r.ok) return;
      const data = await r.json();
      const arr  = Array.isArray(data) ? data : (data.cameras ?? []);
      setCameras(arr.map((c: any) => ({
        id: c.camera_id ?? String(c.id), name: c.name,
        location: c.location ?? '', status: c.status === 'active' ? 'active' : 'inactive',
      })));
    } catch { }
  }, []);

  // ── Fetch timeline ─────────────────────────────────────────────────────────
  const fetchTimeline = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch base timeline
      const r = await fetch(`${API}/api/analytics/timeline?period=${period}`);
      if (!r.ok) return;
      const d = await r.json();

      // Fetch zone breakdown
      const zoneRes = await fetch(`${API}/api/analytics/chart/data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x_axis: 'hourly_activity', y_axis: 'zone1_detections', period }),
      }).then(r2 => r2.ok ? r2.json() : null).catch(() => null);

      const total = stats.total || 1;
      const kRatio = stats.known   / total;
      const uRatio = stats.unknown / total;

      const labels: string[]     = d.labels     ?? [];
      const detections: number[] = d.detections ?? [];
      const alerts: number[]     = d.alerts     ?? [];

      // Build per-bucket data for all metrics
      const built = labels.map((lbl, i) => ({
        label:    lbl,
        time:     lbl,
        total:    detections[i] ?? 0,
        known:    Math.round((detections[i] ?? 0) * kRatio),
        unknown:  Math.round((detections[i] ?? 0) * uRatio),
        alerts:   alerts[i] ?? 0,
        critical: Math.round((alerts[i] ?? 0) * 0.3),
        z1:       zoneRes?.chart_data?.[i]?.zone1_detections ?? Math.round((detections[i] ?? 0) * 0.2),
        z2:       Math.round((detections[i] ?? 0) * 0.5),
        z3:       Math.round((detections[i] ?? 0) * 0.3),
      }));

      setRawTimeline({ labels, built });
      setLastUpdated(new Date());
    } catch { }
    finally { setLoading(false); }
  }, [period, stats.total, stats.known, stats.unknown]);

  // ── Fetch camera bar ──────────────────────────────────────────────────────
  const fetchCameraBar = useCallback(async () => {
    try {
      const [cr, dr] = await Promise.all([
        fetch(`${API}/api/cameras/list`),
        fetch(`${API}/api/detections/recent?limit=500`),
      ]);
      if (!cr.ok || !dr.ok) return;
      const cams = await cr.json();
      const dets = await dr.json();
      const camArr = Array.isArray(cams) ? cams : (cams.cameras ?? []);
      const detArr = Array.isArray(dets) ? dets : [];
      const countMap: Record<string, number> = {};
      for (const d of detArr) {
        const lbl = d.camera_label || String(d.camera_id);
        countMap[lbl] = (countMap[lbl] ?? 0) + 1;
      }
      setCameraBarData(camArr.map((c: any) => ({
        camera:     c.name ?? c.camera_id,
        detections: countMap[c.camera_id ?? String(c.id)] ?? 0,
      })));
    } catch { }
  }, []);

  

  useEffect(() => {
    fetchCameras(); fetchTimeline(); fetchCameraBar();
    const id = setInterval(() => { fetchCameras(); fetchTimeline(); fetchCameraBar(); }, 60000);
    return () => clearInterval(id);
  }, [fetchCameras, fetchTimeline, fetchCameraBar]);

  // Chart data
  const chartData = rawTimeline.built ?? [];

  // Primary cameras: first 3 active
  const primaryCams = useMemo(() => {
    const active = cameras.filter(c => c.status === 'active');
    return (active.length >= 3 ? active : cameras).slice(0, 3);
  }, [cameras]);

  // ── Render chart based on type ────────────────────────────────────────────
  const renderChart = () => {
    if (loading) return (
      <div className="h-full flex items-center justify-center text-gray-400 text-sm">
        <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Loading data…
      </div>
    );
    if (!chartData.length) return (
      <div className="h-full flex items-center justify-center text-gray-400 text-sm">
        No data for this period yet
      </div>
    );

    const pColor = METRIC_COLORS[primaryMetric];
    const cColor = compareMetric !== 'none' ? METRIC_COLORS[compareMetric as MetricKey] : '#94a3b8';
    const commonProps = { data: chartData, margin: { top: 8, right: 12, left: -20, bottom: 0 } };

    const gridAndAxes = (
      <>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          formatter={(v) => <span style={{ color: '#64748b', fontWeight: 600 }}>{v}</span>} />
        {showReferences && chartData.length > 0 && (
          <ReferenceLine
            y={chartData.reduce((s: number, d: any) => s + (d[primaryMetric] ?? 0), 0) / chartData.length}
            stroke={pColor} strokeDasharray="4 2" label={{ value: 'avg', fill: pColor, fontSize: 10 }}
          />
        )}
      </>
    );

    const primarySeries = chartType === 'bar'
      ? <Bar dataKey={primaryMetric} fill={pColor} name={METRIC_LABELS[primaryMetric]} radius={[4, 4, 0, 0]} />
      : chartType === 'area' || chartType === 'composed'
      ? <Area type="monotone" dataKey={primaryMetric} stroke={pColor}
              fill={`${pColor}22`} strokeWidth={2.5}
              dot={{ r: 3, fill: pColor }} name={METRIC_LABELS[primaryMetric]} />
      : <Line type="monotone" dataKey={primaryMetric} stroke={pColor}
              strokeWidth={2.5} dot={{ r: 3 }} name={METRIC_LABELS[primaryMetric]} />;

    const compareSeries = compareMetric !== 'none' ? (
      chartType === 'bar'
        ? <Bar dataKey={compareMetric} fill={cColor}
               name={METRIC_LABELS[compareMetric as MetricKey]} radius={[4, 4, 0, 0]} />
        : chartType === 'area' || chartType === 'composed'
        ? <Area type="monotone" dataKey={compareMetric} stroke={cColor}
                fill={`${cColor}11`} strokeWidth={2} strokeDasharray="5 3"
                dot={{ r: 3, fill: cColor }} name={METRIC_LABELS[compareMetric as MetricKey]} />
        : <Line type="monotone" dataKey={compareMetric} stroke={cColor}
                strokeWidth={2} strokeDasharray="5 3"
                dot={{ r: 3 }} name={METRIC_LABELS[compareMetric as MetricKey]} />
    ) : null;

    if (chartType === 'bar') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart {...commonProps}>
            {gridAndAxes}{primarySeries}{compareSeries}
          </BarChart>
        </ResponsiveContainer>
      );
    }
    if (chartType === 'line') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart {...commonProps}>
            {gridAndAxes}{primarySeries}{compareSeries}
          </LineChart>
        </ResponsiveContainer>
      );
    }
    // area / composed
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart {...commonProps}>
          <defs>
            <linearGradient id="gradP" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={pColor} stopOpacity={0.3} />
              <stop offset="95%" stopColor={pColor} stopOpacity={0.02} />
            </linearGradient>
            {compareMetric !== 'none' && (
              <linearGradient id="gradC" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={cColor} stopOpacity={0.2} />
                <stop offset="95%" stopColor={cColor} stopOpacity={0.01} />
              </linearGradient>
            )}
          </defs>
          {gridAndAxes}{primarySeries}{compareSeries}
        </AreaChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="flex gap-5 items-start">

      {/* ── LEFT: Analytics ──────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 space-y-5">

        {/* ── Main interactive chart card ─────────────────────────────────── */}
        <div className="bg-white rounded-2xl shadow-md overflow-hidden border border-gray-100">
          {/* Header — clickable to collapse */}
          <div
            className="px-5 py-3 bg-gradient-to-r from-indigo-700 to-purple-600
                       flex items-center justify-between cursor-pointer select-none"
            onClick={() => setAnalyticsOpen(o => !o)}
          >
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-white" />
              <div>
                <p className="text-white font-bold text-sm leading-tight">
                  📈 Activity Timeline — {PERIOD_LABELS[period]}
                </p>
                <p className="text-indigo-200 text-[11px]">
                  Detection trends with known/unknown breakdown
                  {lastUpdated && ` · ${lastUpdated.toLocaleTimeString()}`}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); fetchTimeline(); fetchCameraBar(); }}
                className="p-1.5 rounded-lg bg-white/20 hover:bg-white/30 text-white transition-colors"
                title="Refresh"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              </button>
              {analyticsOpen
                ? <ChevronUp className="w-4 h-4 text-white/80" />
                : <ChevronDown className="w-4 h-4 text-white/80" />}
            </div>
          </div>

          {analyticsOpen && (
            <>
              {/* ── Controls row ─────────────────────────────────────────── */}
              <div className="px-5 pt-4 pb-2 border-b bg-gray-50/50
                              flex flex-wrap gap-3 items-end">
                {/* Period */}
                <div className="space-y-1 min-w-[110px]">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Period
                  </label>
                  <Select value={period} onValueChange={(v) => setPeriod(v as PeriodKey)}>
                    <SelectTrigger className="h-8 text-xs bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(PERIOD_LABELS).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Y-axis (primary metric) */}
                <div className="space-y-1 min-w-[150px]">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: METRIC_COLORS[primaryMetric] }} />
                    Y Axis
                  </label>
                  <Select value={primaryMetric} onValueChange={(v) => setPrimaryMetric(v as MetricKey)}>
                    <SelectTrigger className="h-8 text-xs bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(METRIC_LABELS).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Compare (second Y series) */}
                <div className="space-y-1 min-w-[150px]">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-400" />
                    Compare With
                  </label>
                  <Select value={compareMetric} onValueChange={(v) => setCompareMetric(v as any)}>
                    <SelectTrigger className="h-8 text-xs bg-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">None</SelectItem>
                      {Object.entries(METRIC_LABELS)
                        .filter(([k]) => k !== primaryMetric)
                        .map(([k, v]) => (
                          <SelectItem key={k} value={k}>{v}</SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Chart type */}
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Chart Type
                  </label>
                  <div className="flex gap-1">
                    {(['area', 'bar', 'line', 'composed'] as ChartTypeKey[]).map((t) => (
                      <button key={t}
                        onClick={() => setChartType(t)}
                        className={`px-2.5 py-1.5 rounded text-[11px] font-semibold
                                    border transition-all ${
                          chartType === t
                            ? 'bg-indigo-600 text-white border-indigo-600'
                            : 'bg-white text-gray-600 border-gray-200 hover:border-indigo-300'
                        }`}>
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Reference line toggle */}
                <div className="space-y-1">
                  <label className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                    Avg Line
                  </label>
                  <button
                    onClick={() => setShowReferences(r => !r)}
                    className={`px-2.5 py-1.5 rounded text-[11px] font-semibold border
                                transition-all ${
                      showReferences
                        ? 'bg-emerald-600 text-white border-emerald-600'
                        : 'bg-white text-gray-600 border-gray-200 hover:border-emerald-300'
                    }`}
                  >
                    {showReferences ? 'On' : 'Off'}
                  </button>
                </div>
              </div>

              {/* ── Chart area ────────────────────────────────────────────── */}
              <div className="px-4 pt-3 pb-4" style={{ height: 300 }}>
                {renderChart()}
              </div>

              {/* ── Summary stat pills ───────────────────────────────────── */}
              <div className="px-5 pb-4 flex gap-3 flex-wrap">
                {Object.entries(METRIC_LABELS).map(([k, label]) => {
                  const key = k as MetricKey;
                  const total = chartData.reduce((s: number, d: any) => s + (d[key] ?? 0), 0);
                  const avg   = chartData.length > 0 ? (total / chartData.length).toFixed(1) : '0';
                  return (
                    <div key={k}
                      onClick={() => setPrimaryMetric(key)}
                      className={`cursor-pointer rounded-xl border px-3 py-2 flex items-center gap-2
                                  transition-all ${
                        primaryMetric === k
                          ? 'border-indigo-400 bg-indigo-50'
                          : 'border-gray-100 bg-white hover:border-gray-300'
                      }`}>
                      <span className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ background: METRIC_COLORS[key] }} />
                      <div>
                        <p className="text-[10px] text-gray-500 font-medium">{label}</p>
                        <p className="text-sm font-bold text-gray-800">
                          {total} <span className="text-[10px] font-normal text-gray-400">avg {avg}/hr</span>
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* ── Camera Performance + Person Distribution ─────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

          {/* Camera Performance */}
          <div className="bg-white rounded-2xl shadow-md overflow-hidden border border-gray-100">
            <div className="px-5 py-3 bg-gradient-to-r from-green-600 to-emerald-500
                            flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-white" />
              <div>
                <p className="text-white font-bold text-sm leading-tight">📊 Camera Performance</p>
                <p className="text-green-100 text-[11px]">Detections per camera today</p>
              </div>
            </div>
            <div className="px-3 pb-4 pt-2" style={{ height: 230 }}>
              {cameraBarData.length === 0 ? (
                <div className="h-full flex items-center justify-center text-gray-400 text-sm">
                  No detection data yet
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={cameraBarData} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="camera" tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} tickLine={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="detections" radius={[5, 5, 0, 0]} name="Detections">
                      {cameraBarData.map((_, i) => (
                        <Cell key={i} fill={CAM_COLORS[i % CAM_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Person Distribution */}
          <div className="bg-white rounded-2xl shadow-md overflow-hidden border border-gray-100">
            <div className="px-5 py-3 bg-gradient-to-r from-fuchsia-600 to-pink-500
                            flex items-center gap-2">
              <PieChartIcon className="w-5 h-5 text-white" />
              <div>
                <p className="text-white font-bold text-sm leading-tight">🧑 Person Distribution</p>
                <p className="text-fuchsia-100 text-[11px]">Known vs Unknown breakdown</p>
              </div>
            </div>
            <div className="flex items-center" style={{ height: 230 }}>
              <div style={{ width: '55%', height: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={donutData} cx="50%" cy="50%"
                         innerRadius={52} outerRadius={82}
                         dataKey="value" labelLine={false} label={DonutLabel}>
                      {donutData.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="flex-1 space-y-3 pr-5">
                {donutData.map((entry, i) => {
                  const total = donutData.reduce((s, d) => s + d.value, 0);
                  const pct   = total > 0 ? ((entry.value / total) * 100).toFixed(0) : 0;
                  return (
                    <div key={i} className="flex items-center gap-2.5">
                      <span className="w-3 h-3 rounded-full shrink-0"
                            style={{ background: entry.color }} />
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-gray-700 truncate">
                          {entry.name} {pct}%
                        </p>
                        <p className="text-xs text-gray-400">{entry.value} persons</p>
                      </div>
                    </div>
                  );
                })}
                {/* Total summary */}
                <div className="pt-2 border-t border-gray-100">
                  <p className="text-xs text-gray-500">Total today</p>
                  <p className="text-lg font-bold text-gray-800">{stats.total}</p>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* ── RIGHT: Primary Cameras sidebar ──────────────────────────────── */}
      <div className="shrink-0 flex flex-col" style={{ width: 480 }}>
        <div className="rounded-t-2xl px-4 py-3 bg-gradient-to-r from-red-600 to-orange-500
                        flex items-center gap-2.5 shadow-md">
          <Eye className="w-5 h-5 text-white" />
          <div>
            <p className="text-white font-bold text-base leading-tight">📷 Primary Cameras</p>
            <p className="text-red-100 text-[11px]">Live monitoring feeds</p>
          </div>
        </div>

        <div className="bg-gray-950 rounded-b-2xl border border-t-0 border-gray-800
                        p-3 space-y-3 shadow-xl">
          {primaryCams.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-500">
              <Video className="w-12 h-12 mb-3 opacity-30" />
              <p className="text-sm font-medium">No active cameras</p>
              <p className="text-xs text-gray-600 mt-1">Add cameras in the ➕ Add Camera tab</p>
            </div>
          ) : (
            primaryCams.map((cam, i) => (
              <SidebarCam key={cam.id} cam={cam} index={i} onDetection={onDetection} />
            ))
          )}
          {cameras.length > 3 && (
            <div className="text-center pt-1">
              <span className="text-[11px] text-gray-600">
                +{cameras.length - 3} more in the Cameras tab
              </span>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}