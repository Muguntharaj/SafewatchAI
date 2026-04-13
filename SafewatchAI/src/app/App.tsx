import { useState } from 'react';
import { LoginPage } from './components/LoginPage';
import { Dashboard } from './components/Dashboard';

export default function App() {
  const [user, setUser] = useState<{ username: string; role: string } | null>(null);

  const handleLogin = (username: string, role: string) => {
    setUser({ username, role });
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <Dashboard
      username={user.username}
      role={user.role}
      onLogout={handleLogout}
    />
  );
}


// // import { useState } from 'react';
// // import { LoginPage } from './components/LoginPage';
// // import { Dashboard } from './components/Dashboard';
// // import { Toaster } from './components/ui/sonner';

// // export default function App() {
// //   const [isAuthenticated, setIsAuthenticated] = useState(false);
// //   const [user, setUser] = useState<{ username: string; role: string } | null>(null);

// //   const handleLogin = (username: string, role: string) => {
// //     setUser({ username, role });
// //     setIsAuthenticated(true);
// //   };

// //   const handleLogout = () => {
// //     setUser(null);
// //     setIsAuthenticated(false);
// //   };

// //   return (
// //     <>
// //       {!isAuthenticated ? (
// //         <LoginPage onLogin={handleLogin} />
// //       ) : (
// //         <Dashboard
// //           username={user!.username}
// //           role={user!.role}
// //           onLogout={handleLogout}
// //         />
// //       )}
// //       <Toaster />
// //     </>
// //   );
// // }
// /**
//  * App.tsx  — SafeWatch AI central hub
//  *
//  * FIXES / WIRING:
//  *  1. StatsCards – polls GET /api/analytics/dashboard every 10 s (was hardcoded zeros)
//  *  2. TimeLogTable – polls GET /api/detections/recent every 15 s and maps to TimeLog shape
//  *     (was empty array, never fetched)
//  *  3. SuspiciousActivityAlert popup – wired to real alerts from
//  *     GET /api/alerts/list polled every 10 s; new critical alerts auto-open the dialog
//  *  4. PersonClassifier "View Details" – opens classifier with real unknown-person image
//  *     fetched from /media/faces/...  (was showing a grey User icon)
//  *  5. Alert video – onDownloadVideo now opens the real clip URL
//  *     GET /media/alerts/{alert_id}_clip.mp4  (was creating a fake text Blob)
//  *  6. NotificationSlider – unknownPersons fed from GET /api/faces/unknown/list
//  *     so "classify" works against the real /api/faces/unknown/{id}/classify endpoint
//  *  7. WebSocket /ws/live – receives live detections and updates timelog in real-time
//  *  8. RTSP / multiple cameras – CCTVFeed uses  /api/stream/{cameraId}  which the
//  *     backend serves as MJPEG; cameras are loaded from GET /api/cameras/list
//  */

// import { useState, useEffect, useRef, useCallback } from 'react';
// import { StatsCards } from './components/StatsCards';
// import { TimeLogTable, TimeLog } from './components/TimeLogTable';
// import { PersonClassifier, Person } from './components/PersonClassifier';
// import { SuspiciousActivityAlert, SuspiciousActivity } from './components/SuspiciousActivityAlert';
// import { NotificationSlider } from './components/NotificationSlider';
// import { HomePage } from './components/HomePage';
// import { AddCamera } from './components/AddCamera';
// import { toast } from 'sonner';

// // ── Types ─────────────────────────────────────────────────────────────────────
// interface Stats {
//   total: number;
//   known: number;
//   unknown: number;
//   currentlyIn: number;
//   alerts: number;
// }

// interface ApiAlert {
//   id: number;
//   alert_id: string;
//   alert_type: string;
//   alert_level: number;
//   severity: string;
//   camera_id: number | null;
//   zone: number;
//   description: string;
//   action_detected: string;
//   video_path: string | null;
//   snapshot_path: string | null;
//   status: string;
//   created_at: string;
// }

// interface ApiDetection {
//   id: number;
//   camera_id: number;
//   action: string | null;
//   action_confidence: number | null;
//   confidence: number;
//   zone: number;
//   timestamp: string;
// }

// interface ApiUnknown {
//   id: number;
//   face_image_path: string;
//   camera_id: number;
//   detection_time: string;
// }

// const API = 'http://localhost:8000';

// // ── Helpers ───────────────────────────────────────────────────────────────────
// function alertToPerson(u: ApiUnknown, index: number): Person {
//   // Convert file-system path → URL served by /media/faces/...
//   const filename = u.face_image_path.split(/[\\/]/).pop() ?? '';
//   return {
//     id: String(u.id),
//     category: 'Unknown',
//     confidence: 1,
//     timestamp: new Date(u.detection_time),
//     // FIX #4: real face image URL instead of grey placeholder
//     imageData: `${API}/media/faces/unknown/${filename}`,
//   };
// }

// function apiAlertToActivity(a: ApiAlert): SuspiciousActivity {
//   const typeMap: Record<string, SuspiciousActivity['type']> = {
//     fighting: 'fighting', loitering: 'loitering', theft: 'theft',
//     vandalism: 'vandalism', trespassing: 'trespassing',
//     weapon_detected: 'weapon', weapon: 'weapon',
//   };
//   return {
//     id: a.alert_id,
//     cameraId: String(a.camera_id ?? ''),
//     cameraName: `Camera ${a.camera_id ?? '?'}`,
//     type: typeMap[a.alert_type] ?? 'loitering',
//     severity: (a.severity as SuspiciousActivity['severity']) ?? 'medium',
//     timestamp: new Date(a.created_at),
//     description: a.description,
//     // FIX #5: real clip URL
//     videoClip: a.video_path
//       ? `${API}/media/alerts/${a.alert_id}_clip.mp4`
//       : undefined,
//   };
// }

// function detectionToLog(d: ApiDetection, index: number): TimeLog {
//   return {
//     id: String(d.id),
//     personId: `cam-${d.camera_id}-det-${d.id}`,
//     // FIX #2: show actual action as person name when no face recognised
//     personName: d.action ? `Action: ${d.action}` : 'Person Detected',
//     // FIX #2: map action_confidence → category
//     category: d.action && d.action !== 'normal' ? 'Unknown' : 'Employee',
//     inTime: new Date(d.timestamp),
//     outTime: undefined,
//     status: 'In',
//     cameraId: `cam-${d.camera_id}`,
//     cameraName: `Camera ${d.camera_id}`,
//   };
// }

// // ─────────────────────────────────────────────────────────────────────────────
// export default function App() {
//   // ── State ─────────────────────────────────────────────────────────────────
//   const [stats, setStats] = useState<Stats>({
//     total: 0, known: 0, unknown: 0, currentlyIn: 0, alerts: 0,
//   });
//   const [timeLogs, setTimeLogs]           = useState<TimeLog[]>([]);
//   const [unknownPersons, setUnknownPersons] = useState<Person[]>([]);
//   const [suspiciousActivities, setSuspiciousActivities] = useState<SuspiciousActivity[]>([]);

//   const [selectedPerson, setSelectedPerson] = useState<Person | null>(null);
//   const [showPersonClassifier, setShowPersonClassifier] = useState(false);

//   const [selectedActivity, setSelectedActivity] = useState<SuspiciousActivity | null>(null);
//   const [showActivityAlert, setShowActivityAlert] = useState(false);

//   const [showNotificationSlider, setShowNotificationSlider] = useState(false);
//   const [notificationTab, setNotificationTab] = useState<'alerts' | 'unknown' | 'recent'>('alerts');

//   const [activeTab, setActiveTab] = useState<'home' | 'cameras' | 'addcamera'>('home');

//   const wsRef   = useRef<WebSocket | null>(null);
//   const seenAlerts = useRef<Set<string>>(new Set());

//   // ── Fetch stats ───────────────────────────────────────────────────────────
//   const fetchStats = useCallback(async () => {
//     try {
//       const r = await fetch(`${API}/api/analytics/dashboard?period=today`);
//       if (!r.ok) return;
//       const data = await r.json();
//       setStats({
//         total:      data.total_detections  ?? 0,
//         known:      data.known_persons      ?? 0,
//         unknown:    data.unknown_persons    ?? 0,
//         currentlyIn: data.currently_in     ?? 0,
//         alerts:     data.total_alerts       ?? 0,
//       });
//     } catch (e) { /* backend not up yet */ }
//   }, []);

//   // ── Fetch recent detections → timelog ────────────────────────────────────
//   const fetchTimeLogs = useCallback(async () => {
//     try {
//       const r = await fetch(`${API}/api/detections/recent?limit=50`);
//       if (!r.ok) return;
//       const data: ApiDetection[] = await r.json();
//       setTimeLogs(data.map(detectionToLog));
//     } catch (e) {}
//   }, []);

//   // ── Fetch alerts ──────────────────────────────────────────────────────────
//   const fetchAlerts = useCallback(async () => {
//     try {
//       const r = await fetch(`${API}/api/alerts/list?limit=50`);
//       if (!r.ok) return;
//       const data: ApiAlert[] = await r.json();
//       const activities = data.map(apiAlertToActivity);
//       setSuspiciousActivities(activities);

//       // FIX #3: auto-popup for new critical / high alerts
//       for (const a of data) {
//         if (!seenAlerts.current.has(a.alert_id)) {
//           seenAlerts.current.add(a.alert_id);
//           if (a.alert_level <= 2 && a.status === 'active') {
//             const activity = apiAlertToActivity(a);
//             setSelectedActivity(activity);
//             setShowActivityAlert(true);
//             toast.error(`🚨 ${a.severity.toUpperCase()} Alert: ${a.alert_type}`, {
//               duration: 6000,
//             });
//           }
//         }
//       }
//     } catch (e) {}
//   }, []);

//   // ── Fetch unknown persons ─────────────────────────────────────────────────
//   const fetchUnknown = useCallback(async () => {
//     try {
//       const r = await fetch(`${API}/api/faces/unknown/list?limit=50`);
//       if (!r.ok) return;
//       const data: ApiUnknown[] = await r.json();
//       setUnknownPersons(data.map(alertToPerson));
//     } catch (e) {}
//   }, []);

//   // ── WebSocket ─────────────────────────────────────────────────────────────
//   useEffect(() => {
//     function connect() {
//       const ws = new WebSocket(`ws://localhost:8000/ws/live`);
//       wsRef.current = ws;

//       ws.onmessage = (evt) => {
//         try {
//           const msg = JSON.parse(evt.data);
//           if (msg.type === 'detections' && Array.isArray(msg.detections)) {
//             // FIX #7: update timelog in real-time from WS events
//             const now = new Date();
//             const newLogs: TimeLog[] = msg.detections.map((d: any, i: number) => ({
//               id: `ws-${Date.now()}-${i}`,
//               personId: `ws-${msg.camera_id}-${i}`,
//               personName: d.person?.name ?? (d.action !== 'normal' ? `Action: ${d.action}` : 'Person Detected'),
//               category: d.person?.classification === 'unknown' ? 'Unknown' : 'Employee',
//               inTime: now,
//               status: 'In' as const,
//               cameraId: msg.camera_id,
//               cameraName: `Camera ${msg.camera_id}`,
//             }));
//             if (newLogs.length > 0) {
//               setTimeLogs(prev => [...newLogs, ...prev].slice(0, 200));
//             }
//           }
//         } catch {}
//       };

//       ws.onclose = () => {
//         setTimeout(connect, 3000);   // auto-reconnect
//       };
//     }
//     connect();
//     return () => wsRef.current?.close();
//   }, []);

//   // ── Polling ───────────────────────────────────────────────────────────────
//   useEffect(() => {
//     fetchStats();
//     fetchTimeLogs();
//     fetchAlerts();
//     fetchUnknown();

//     const intervals = [
//       setInterval(fetchStats,    10_000),
//       setInterval(fetchTimeLogs, 15_000),
//       setInterval(fetchAlerts,   10_000),
//       setInterval(fetchUnknown,  20_000),
//     ];
//     return () => intervals.forEach(clearInterval);
//   }, [fetchStats, fetchTimeLogs, fetchAlerts, fetchUnknown]);

//   // ── Handlers ──────────────────────────────────────────────────────────────
//   const handleViewPerson = (log: TimeLog) => {
//     // Try to find matching unknown person by cameraId / time
//     const match = unknownPersons.find(p => p.cameraId === log.cameraId) ?? {
//       id: log.personId,
//       category: log.category as Person['category'],
//       confidence: 1,
//       timestamp: log.inTime,
//     };
//     setSelectedPerson(match);
//     setShowPersonClassifier(true);
//   };

//   // FIX #6: classify calls real API
//   const handleClassifyPerson = async (id: string, name: string, category: string) => {
//     try {
//       await fetch(`${API}/api/faces/unknown/${id}/classify`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ name, classification: category.toLowerCase() }),
//       });
//       toast.success(`✅ ${name} classified as ${category}`);
//       fetchUnknown();
//       fetchStats();
//     } catch {
//       toast.error('Failed to classify person');
//     }
//   };

//   // FIX #5: open real alert video URL in a new tab
//   const handleDownloadVideo = (activity: SuspiciousActivity) => {
//     if (activity.videoClip) {
//       window.open(activity.videoClip, '_blank');
//     } else {
//       toast.error('No video clip available for this alert yet');
//     }
//   };

//   const handleDownloadImage = (person: Person) => {
//     if (person.imageData) {
//       const a = document.createElement('a');
//       a.href = person.imageData;
//       a.download = `unknown_${person.id}.jpg`;
//       a.click();
//     }
//   };

//   const handleExportLogs = () => {
//     const csv = [
//       'Name,Category,Camera,In Time,Status',
//       ...timeLogs.map(l =>
//         `${l.personName},${l.category},${l.cameraName},${l.inTime.toLocaleString()},${l.status}`
//       ),
//     ].join('\n');
//     const blob = new Blob([csv], { type: 'text/csv' });
//     const url  = URL.createObjectURL(blob);
//     const a    = document.createElement('a');
//     a.href = url; a.download = 'timelog.csv'; a.click();
//     URL.revokeObjectURL(url);
//   };

//   const handleStatsCardClick = (type: 'total' | 'known' | 'unknown' | 'active' | 'alerts') => {
//     if (type === 'alerts') {
//       setNotificationTab('alerts');
//       setShowNotificationSlider(true);
//     } else if (type === 'unknown') {
//       setNotificationTab('unknown');
//       setShowNotificationSlider(true);
//     } else {
//       setNotificationTab('recent');
//       setShowNotificationSlider(true);
//     }
//   };

//   // ── Render ────────────────────────────────────────────────────────────────
//   return (
//     <div className="min-h-screen bg-gray-950 text-white">
//       {/* ── Top nav ── */}
//       <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between">
//         <div className="flex items-center gap-3">
//           <span className="text-2xl">🛡️</span>
//           <span className="font-bold text-xl">SafeWatch AI</span>
//           <span className="text-xs text-gray-400 ml-2">by Neelaminds</span>
//         </div>
//         <div className="flex gap-2">
//           {(['home', 'cameras', 'addcamera'] as const).map(tab => (
//             <button
//               key={tab}
//               onClick={() => setActiveTab(tab)}
//               className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
//                 activeTab === tab
//                   ? 'bg-blue-600 text-white'
//                   : 'text-gray-400 hover:text-white hover:bg-gray-800'
//               }`}
//             >
//               {tab === 'home' ? '🏠 Dashboard' : tab === 'cameras' ? '📹 Cameras' : '➕ Add Camera'}
//             </button>
//           ))}
//           <button
//             onClick={() => setShowNotificationSlider(true)}
//             className="relative px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800"
//           >
//             🔔 Alerts
//             {stats.alerts > 0 && (
//               <span className="absolute -top-1 -right-1 bg-red-600 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center">
//                 {stats.alerts > 9 ? '9+' : stats.alerts}
//               </span>
//             )}
//           </button>
//         </div>
//       </nav>

//       {/* ── Main ── */}
//       <main className="p-6 space-y-6">
//         {/* Stats always visible */}
//         <StatsCards stats={stats} onCardClick={handleStatsCardClick} />

//         {activeTab === 'home' && (
//           <HomePage stats={stats} />
//         )}

//         {activeTab === 'cameras' && (
//           <div className="space-y-6">
//             {/* Time Log */}
//             <TimeLogTable
//               logs={timeLogs}
//               onViewPerson={handleViewPerson}
//               onExport={handleExportLogs}
//             />
//           </div>
//         )}

//         {activeTab === 'addcamera' && (
//           <AddCamera onCameraAdded={() => { setActiveTab('cameras'); fetchStats(); }} />
//         )}
//       </main>

//       {/* ── Modals & Sliders ── */}

//       {/* Person Classifier */}
//       {selectedPerson && (
//         <PersonClassifier
//           person={selectedPerson}
//           isOpen={showPersonClassifier}
//           onClose={() => setShowPersonClassifier(false)}
//           onClassify={handleClassifyPerson}
//           onDownload={handleDownloadImage}
//         />
//       )}

//       {/* Suspicious Activity Popup */}
//       {selectedActivity && (
//         <SuspiciousActivityAlert
//           activity={selectedActivity}
//           isOpen={showActivityAlert}
//           onClose={() => setShowActivityAlert(false)}
//           onDownloadVideo={handleDownloadVideo}
//         />
//       )}

//       {/* Notification Slider */}
//       <NotificationSlider
//         isOpen={showNotificationSlider}
//         onClose={() => setShowNotificationSlider(false)}
//         unknownPersons={unknownPersons}
//         suspiciousActivities={suspiciousActivities}
//         recentLogs={timeLogs.slice(0, 20)}
//         onClassifyPerson={(person) => {
//           setSelectedPerson(person);
//           setShowPersonClassifier(true);
//         }}
//         onDownloadImage={handleDownloadImage}
//         onDownloadVideo={handleDownloadVideo}
//         initialTab={notificationTab}
//       />
//     </div>
//   );
// }
