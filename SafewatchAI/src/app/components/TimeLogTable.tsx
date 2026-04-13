/**
 * TimeLogTable.tsx — v2 Bug-Fixed Build
 *
 * FIXES IN THIS VERSION:
 * ──────────────────────
 *
 * 1. VIEW BUTTON DOESN'T SHOW UNKNOWN PERSON IMAGE
 *    Root cause 1: FaceThumbnail used a `useEffect` but called `useState`
 *    without importing `useEffect` — it was missing from imports.
 *    Root cause 2: handleViewPerson was passed `log.imageData` but the
 *    TimeLogTable never passed `imageData` down to PersonClassifier properly
 *    because `onViewPerson` expected only a TimeLog, not a Person.
 *    Fix: FaceThumbnail now has both useState + useEffect imported.
 *    Fix: The imageData in the log is now correctly forwarded via
 *    the onViewPerson callback. Dashboard.handleViewPerson already has
 *    the fallback chain — we just ensure imageData is populated.
 *
 * 2. NOT GETTING NEW DATA
 *    Root cause: The table had no manual refresh control — it relied
 *    entirely on the 20-second poll in Dashboard. Added a "Refresh" button
 *    and an `onRefresh` optional prop. When clicked it calls onRefresh
 *    immediately (Dashboard passes fetchLogs + fetchUnknown).
 *    Also added a "Last updated" timestamp display.
 *
 * 3. FILTER UNKNOWN PERSON IMAGE IN VIEW
 *    The FaceThumbnail component now tries the known/ folder as fallback
 *    if the unknown/ URL fails (person may have been classified since detection).
 *
 * 4. DATE FILTER for TimeLogTable
 *    Added "Today / Yesterday / Last 7 days" segmented control at the top.
 *    Default: Today. All filtering is pure-frontend, no extra API calls.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Table, TableBody, TableCell, TableHead,
  TableHeader, TableRow,
} from './ui/table';
import { Badge }   from './ui/badge';
import { Button }  from './ui/button';
import { Download, Clock, UserCheck, UserX, Eye, X, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

const API = 'http://localhost:8000';

export interface TimeLog {
  id:         string;
  personId:   string;
  personName: string;
  category:   'Employee' | 'Owner' | 'Unknown';
  inTime:     Date;
  outTime?:   Date;
  status:     'In' | 'Out';
  cameraId:   string;
  cameraName: string;
  imageData?: string;
}

interface TimeLogTableProps {
  logs:         TimeLog[];
  onViewPerson: (log: TimeLog) => void;
  onExport:     () => void;
  onRefresh?:   () => void;   // NEW: called when user clicks Refresh button
}

// ── Face thumbnail — fixed imports ────────────────────────────────────────────
function FaceThumbnail({ imageData, name }: { imageData?: string; name: string }) {
  const [failed, setFailed] = useState(false);
  const [src,    setSrc]    = useState<string | null>(null);

  // FIX: useEffect was missing from import in original
  useEffect(() => {
    if (!imageData) { setSrc(null); return; }
    if (imageData.startsWith('http')) { setSrc(imageData); return; }
    const fn = imageData.split(/[/\\]/).pop();
    if (fn && fn.includes('.')) {
      setSrc(`${API}/media/faces/unknown/${fn}`);
    } else {
      setSrc(null);
    }
    setFailed(false);
  }, [imageData]);

  // FIX: Try known/ folder as fallback when unknown/ fails
  const handleError = useCallback(() => {
    if (src && src.includes('/unknown/')) {
      const knownUrl = src.replace('/unknown/', '/known/');
      setSrc(knownUrl);
    } else {
      setFailed(true);
    }
  }, [src]);

  if (!src || failed) {
    const initials = name
      .split(' ')
      .map(w => w[0] ?? '')
      .join('')
      .slice(0, 2)
      .toUpperCase() || '?';
    return (
      <div style={{
        width: 30, height: 30, borderRadius: 8, flexShrink: 0,
        background: '#e0e7ef',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700, color: '#64748b', userSelect: 'none',
      }}>
        {initials}
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={name}
      onError={handleError}
      style={{
        width: 30, height: 30, borderRadius: 8, flexShrink: 0,
        objectFit: 'cover',
        border: '1.5px solid #e2e8f0',
      }}
    />
  );
}

// ── Date filter type ──────────────────────────────────────────────────────────
type DateFilter = 'today' | 'yesterday' | 'last7' | 'all';

function getFilterRange(f: DateFilter): { from: Date; to: Date } | null {
  const now = new Date();
  if (f === 'all') return null;
  if (f === 'today') {
    const from = new Date(now); from.setHours(0, 0, 0, 0);
    const to   = new Date(now); to.setHours(23, 59, 59, 999);
    return { from, to };
  }
  if (f === 'yesterday') {
    const from = new Date(now.getTime() - 86_400_000); from.setHours(0, 0, 0, 0);
    const to   = new Date(from); to.setHours(23, 59, 59, 999);
    return { from, to };
  }
  if (f === 'last7') {
    const from = new Date(now.getTime() - 7 * 86_400_000);
    return { from, to: new Date(now.getTime() + 60_000) };
  }
  return null;
}

function matchesDateFilter(log: TimeLog, f: DateFilter): boolean {
  const range = getFilterRange(f);
  if (!range) return true;
  return log.inTime >= range.from && log.inTime <= range.to;
}

// ── Main ──────────────────────────────────────────────────────────────────────
export function TimeLogTable({ logs, onViewPerson, onExport, onRefresh }: TimeLogTableProps) {
  const [loadingId,      setLoadingId]      = useState<string | null>(null);
  const [filterPersonId, setFilterPersonId] = useState<string | null>(null);
  const [filterName,     setFilterName]     = useState('');
  const [dateFilter,     setDateFilter]     = useState<DateFilter>('today');
  const [lastUpdated,    setLastUpdated]    = useState<Date>(new Date());
  const [isRefreshing,   setIsRefreshing]   = useState(false);

  // Update lastUpdated whenever logs changes
  useEffect(() => { setLastUpdated(new Date()); }, [logs.length]);

  const handleView = (log: TimeLog) => {
    setLoadingId(log.id);
    setTimeout(() => { onViewPerson(log); setLoadingId(null); }, 80);
  };

  const handleRefresh = async () => {
    if (!onRefresh) return;
    setIsRefreshing(true);
    onRefresh();
    setTimeout(() => setIsRefreshing(false), 1200);
  };

  const togglePersonFilter = (log: TimeLog, e: React.MouseEvent) => {
    e.stopPropagation();
    if (filterPersonId === log.personId) {
      setFilterPersonId(null); setFilterName('');
    } else {
      setFilterPersonId(log.personId); setFilterName(log.personName);
    }
  };

  // Apply date filter first, then person filter
  const dateFiltered = logs.filter(l => matchesDateFilter(l, dateFilter));
  const displayedLogs = filterPersonId
    ? dateFiltered.filter(l =>
        l.personId === filterPersonId ||
        (l.personName === filterName && filterName !== 'Unknown Person')
      )
    : dateFiltered;

  const getCategoryColor = (c: string) =>
    c === 'Owner'    ? 'bg-purple-100 text-purple-800 border-purple-200'
    : c === 'Employee' ? 'bg-blue-100 text-blue-800 border-blue-200'
                       : 'bg-orange-100 text-orange-800 border-orange-200';

  const getStatusColor = (s: string) =>
    s === 'In'
      ? 'bg-green-100 text-green-800 border-green-200'
      : 'bg-gray-100 text-gray-800 border-gray-200';

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />Visitor Time Tracking
            </CardTitle>
            <CardDescription>
              Complete entry &amp; exit records · click a name to filter by person
              <span className="ml-2 text-[10px] text-slate-400">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            </CardDescription>
          </div>
          <div className="flex gap-2 items-center flex-wrap">
            {onRefresh && (
              <Button
                onClick={handleRefresh}
                variant="outline"
                size="sm"
                className="gap-2 text-xs"
                disabled={isRefreshing}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
                {isRefreshing ? 'Refreshing…' : 'Refresh'}
              </Button>
            )}
            <Button onClick={onExport} variant="outline" size="sm" className="gap-2 text-xs">
              <Download className="w-4 h-4" />Export
            </Button>
          </div>
        </div>

        {/* Date filter segmented control */}
        <div className="flex items-center gap-1.5 mt-2">
          {(['today', 'yesterday', 'last7', 'all'] as DateFilter[]).map(f => (
            <button
              key={f}
              onClick={() => setDateFilter(f)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all border ${
                dateFilter === f
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
              }`}
            >
              {f === 'today' ? 'Today'
               : f === 'yesterday' ? 'Yesterday'
               : f === 'last7' ? 'Last 7 days'
               : 'All time'}
            </button>
          ))}
          <span className="text-xs text-slate-400 ml-1">
            ({dateFiltered.length} record{dateFiltered.length !== 1 ? 's' : ''})
          </span>
        </div>

        {/* Active person filter chip */}
        {filterPersonId && (
          <div className="flex items-center gap-2 mt-2 px-3 py-1.5
                          bg-blue-50 border border-blue-200 rounded-lg w-fit text-sm">
            <span className="text-blue-700 font-medium">
              Filtering: <strong>{filterName}</strong>
            </span>
            <span className="text-blue-400 text-xs">
              ({displayedLogs.length} record{displayedLogs.length !== 1 ? 's' : ''})
            </span>
            <button
              onClick={() => { setFilterPersonId(null); setFilterName(''); }}
              className="text-blue-400 hover:text-blue-600 ml-1"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <div className="rounded-md border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-gray-50">
                <TableHead className="w-10">Face</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Camera</TableHead>
                <TableHead>In Time</TableHead>
                <TableHead>Out Time</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead className="text-right">Details</TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {displayedLogs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-muted-foreground py-12">
                    <div className="flex flex-col items-center gap-2">
                      <Clock className="w-10 h-10 text-gray-300" />
                      <p>
                        {filterPersonId
                          ? `No visits found for ${filterName}`
                          : dateFilter === 'today'
                          ? 'No visitor logs yet today. Activity will appear here automatically.'
                          : `No visitor logs for the selected period.`}
                      </p>
                      {onRefresh && (
                        <Button variant="outline" size="sm" onClick={handleRefresh}
                          className="mt-2 gap-2 text-xs">
                          <RefreshCw className="w-3.5 h-3.5" />Refresh now
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                displayedLogs.map((log) => {
                  const duration  = log.outTime
                    ? Math.round((log.outTime.getTime() - log.inTime.getTime()) / 60000)
                    : null;
                  const isLoading  = loadingId === log.id;
                  const isFiltered = filterPersonId === log.personId;

                  return (
                    <TableRow
                      key={log.id}
                      className={`cursor-pointer transition-colors ${
                        isFiltered ? 'bg-blue-50/60 hover:bg-blue-50' : 'hover:bg-slate-50'
                      }`}
                      onClick={() => handleView(log)}
                    >
                      {/* Face thumbnail */}
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <FaceThumbnail imageData={log.imageData} name={log.personName} />
                      </TableCell>

                      {/* Name — click to filter */}
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          {log.category === 'Unknown'
                            ? <UserX className="w-3.5 h-3.5 text-orange-500 shrink-0" />
                            : <UserCheck className="w-3.5 h-3.5 text-green-500 shrink-0" />}
                          <button
                            className={`truncate max-w-[120px] text-left transition-colors ${
                              isFiltered
                                ? 'text-blue-700 font-semibold underline'
                                : 'hover:text-blue-600 hover:underline'
                            }`}
                            title="Click to show all visits by this person"
                            onClick={(e) => togglePersonFilter(log, e)}
                          >
                            {log.personName}
                          </button>
                        </div>
                      </TableCell>

                      <TableCell>
                        <Badge variant="secondary"
                          className={`border text-xs ${getCategoryColor(log.category)}`}>
                          {log.category}
                        </Badge>
                      </TableCell>

                      <TableCell className="text-sm text-muted-foreground truncate max-w-[110px]">
                        {log.cameraName}
                      </TableCell>

                      <TableCell className="font-mono text-sm">
                        {log.inTime.toLocaleTimeString()}
                      </TableCell>

                      <TableCell className="font-mono text-sm">
                        {log.outTime ? log.outTime.toLocaleTimeString() : '—'}
                      </TableCell>

                      <TableCell>
                        <Badge variant="secondary"
                          className={`border text-xs ${getStatusColor(log.status)}`}>
                          {log.status === 'In' ? '🟢 In' : '⚫ Out'}
                        </Badge>
                      </TableCell>

                      <TableCell className="text-sm">
                        {duration !== null ? `${duration} min` : '—'}
                      </TableCell>

                      <TableCell className="text-right"
                        onClick={(e) => { e.stopPropagation(); handleView(log); }}>
                        <Button variant="ghost" size="sm" disabled={isLoading}
                          className="gap-1.5 text-blue-600 hover:text-blue-700 hover:bg-blue-50">
                          {isLoading ? (
                            <>
                              <span className="w-3.5 h-3.5 border-2 border-blue-500
                                             border-t-transparent rounded-full animate-spin" />
                              Opening…
                            </>
                          ) : (
                            <><Eye className="w-3.5 h-3.5" />View</>
                          )}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>

        {logs.length > 0 && (
          <p className="text-xs text-gray-400 mt-3 text-right">
            {filterPersonId
              ? `${displayedLogs.length} of ${dateFiltered.length} records (filtered by person)`
              : `Showing ${displayedLogs.length} of ${logs.length} record${logs.length !== 1 ? 's' : ''}`}
          </p>
        )}
      </CardContent>
    </Card>
  );
}