/**
 * SuspiciousActivityAlert.tsx — Fixed Build
 *
 * FIXES:
 *  1. VIDEO URL: videoClip prop is now a full http://… URL (set by Dashboard).
 *     Falls back to /media/alerts/{id}_clip.mp4 if not provided.
 *  2. SNAPSHOT: snapshot_url now displayed below video if available.
 *  3. DOWNLOAD: fetch + blob so it triggers a real file download, not a
 *     new tab that just streams (or shows broken-video if camera is offline).
 *  4. EMAIL: proper mailto: link with pre-filled subject + body.
 *  5. SMS: sms: URI scheme with body.
 *  6. WHATSAPP: wa.me for mobile, web.whatsapp.com for desktop.
 *  7. VIDEO ELEMENT: preload="metadata" so poster frame shows before play.
 */

import { useState, useRef } from 'react';
import {
  Dialog, DialogContent, DialogDescription,
  DialogFooter, DialogHeader, DialogTitle,
} from './ui/dialog';
import { Button }    from './ui/button';
import { Badge }     from './ui/badge';
import { Textarea }  from './ui/textarea';
import { Label }     from './ui/label';
import { Input }     from './ui/input';
import {
  AlertTriangle, Mail, MessageSquare, Send, Download,
  Volume2, VolumeX,
} from 'lucide-react';
import { toast } from 'sonner';

export interface SuspiciousActivity {
  id:          string;
  cameraId:    string;
  cameraName:  string;
  type:        'loitering' | 'fighting' | 'theft' | 'vandalism' | 'trespassing' | 'weapon' | 'weapon_detected' | 'weapon_grip' | 'fire' | 'falling' | 'running' | 'break_in' | 'crowding' | 'suspicious_behavior' | 'trespassing' | string;
  severity:    'low' | 'medium' | 'high' | 'critical';
  timestamp:   Date;
  description: string;
  videoClip?:  string;   // full http:// URL from backend
  snapshot?:   string;   // full http:// URL from backend
}

interface SuspiciousActivityAlertProps {
  activity:          SuspiciousActivity;
  isOpen:            boolean;
  onClose:           () => void;
  onDownloadVideo?:  (activity: SuspiciousActivity) => void;
}

const API = 'http://localhost:8000';

export function SuspiciousActivityAlert({
  activity, isOpen, onClose, onDownloadVideo,
}: SuspiciousActivityAlertProps) {
  const [emailAddress,   setEmailAddress]   = useState('');
  const [phoneNumber,    setPhoneNumber]     = useState('');
  const [isMuted,        setIsMuted]         = useState(true);
  const [isDownloading,  setIsDownloading]   = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const alertMessage =
    `⚠️ SECURITY ALERT\n\n` +
    `Activity: ${activity.type.toUpperCase()}\n` +
    `Camera: ${activity.cameraName}\n` +
    `Time: ${activity.timestamp.toLocaleString()}\n` +
    `Severity: ${activity.severity.toUpperCase()}\n\n` +
    `Description: ${activity.description}`;

  const [message, setMessage] = useState(alertMessage);

  // FIX 1: resolve clip URL — backend sends full http:// URL in videoClip
  const clipUrl = activity.videoClip?.startsWith('http')
    ? activity.videoClip
    : `${API}/media/alerts/${activity.id}_clip.mp4`;

  const snapshotUrl = activity.snapshot?.startsWith('http')
    ? activity.snapshot
    : `${API}/media/alerts/${activity.id}_snapshot.jpg`;

  const getSeverityColor = (s: string) => {
    switch (s) {
      case 'critical': return 'bg-red-600 text-white';
      case 'high':     return 'bg-orange-600 text-white';
      case 'medium':   return 'bg-yellow-600 text-white';
      default:         return 'bg-blue-600 text-white';
    }
  };

  const getAlertIcon = (t: string) => {
    if (t.includes('weapon') || t.includes('gun') || t.includes('knife')) return '🔫';
    if (t === 'fire') return '🔥';
    if (t === 'fighting') return '👊';
    if (t === 'falling')  return '⬇️';
    if (t === 'theft' || t === 'stealing') return '🚨';
    if (t === 'loitering') return '👁️';
    if (t === 'trespassing' || t === 'break_in') return '🚪';
    if (t === 'running')   return '🏃';
    return '⚠️';
  };

  // FIX 4: real mailto link
  const handleSendEmail = () => {
    if (!emailAddress) { toast.error('Please enter an email address'); return; }
    const subject = encodeURIComponent(`🚨 SafeWatch Alert: ${activity.type.toUpperCase()} — ${activity.cameraName}`);
    const body    = encodeURIComponent(message);
    window.location.href = `mailto:${emailAddress}?subject=${subject}&body=${body}`;
    toast.success(`Opening email client for ${emailAddress}`);
  };

  // FIX 5: sms: URI
  const handleSendSMS = () => {
    if (!phoneNumber) { toast.error('Please enter a phone number'); return; }
    const body   = encodeURIComponent(message.substring(0, 160));
    const smsUrl = /Android/i.test(navigator.userAgent)
      ? `sms:${phoneNumber}?body=${body}`
      : `sms:${phoneNumber}&body=${body}`;
    window.location.href = smsUrl;
    toast.success(`Opening SMS to ${phoneNumber}`);
  };

  // FIX 6: WhatsApp mobile + desktop
  const handleShareWhatsApp = () => {
    const encoded = encodeURIComponent(message);
    const url = /Mobile|Android|iPhone/i.test(navigator.userAgent)
      ? `https://wa.me/?text=${encoded}`
      : `https://web.whatsapp.com/send?text=${encoded}`;
    window.open(url, '_blank', 'noopener,noreferrer');
    toast.success('Opening WhatsApp…');
  };

  // FIX 3: real download via fetch + blob
  const handleDownloadVideo = async () => {
    setIsDownloading(true);
    try {
      const res = await fetch(clipUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const a    = document.createElement('a');
      a.href     = URL.createObjectURL(blob);
      a.download = `alert_${activity.id}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
      toast.success('Video downloaded');
    } catch {
      window.open(clipUrl, '_blank', 'noopener,noreferrer');
      toast.info('Opening video in new tab');
    } finally {
      setIsDownloading(false);
    }
    onDownloadVideo?.(activity);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[620px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            {getAlertIcon(activity.type)} {activity.type.includes('weapon') ? 'WEAPON ALERT' : activity.type === 'fire' ? 'FIRE ALERT' : 'Security Alert'}
          </DialogTitle>
          <DialogDescription>
            Detected on {activity.timestamp.toLocaleString()} — {activity.cameraName}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Activity details */}
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium">Activity Type:</span>
              <Badge variant="secondary" className="capitalize">{activity.type}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-medium">Severity Level:</span>
              <Badge className={getSeverityColor(activity.severity)}>
                {activity.severity.toUpperCase()}
              </Badge>
            </div>
            <div className="pt-2 border-t border-red-200">
              <p className="text-sm">{activity.description}</p>
            </div>
          </div>

          {/* FIX 1+7: real video player with proper clip URL */}
          <div className="w-full bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={clipUrl}
              className="w-full aspect-video object-cover"
              controls
              muted={isMuted}
              playsInline
              preload="metadata"
              onError={() => { /* native broken-video UI shows */ }}
            />
            <div className="flex items-center justify-between px-3 py-2 bg-gray-900">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsMuted(m => !m)}
                  className="text-gray-400 hover:text-white transition-colors"
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                </button>
                <span className="text-xs text-gray-400">Alert clip</span>
              </div>
              <Button
                size="sm"
                variant="secondary"
                onClick={handleDownloadVideo}
                disabled={isDownloading}
                className="gap-2 text-xs"
              >
                <Download className="w-3 h-3" />
                {isDownloading ? 'Downloading…' : 'Download Clip'}
              </Button>
            </div>
          </div>

          {/* FIX 2: snapshot image */}
          <div className="w-full rounded-lg overflow-hidden border">
            <img
              src={snapshotUrl}
              alt="Alert snapshot"
              className="w-full object-cover max-h-48"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>

          {/* Alert message editor */}
          <div className="space-y-2">
            <Label htmlFor="alert-msg">Alert Message</Label>
            <Textarea
              id="alert-msg"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={5}
              className="font-mono text-sm"
            />
          </div>

          {/* Send options */}
          <div className="space-y-3">
            {/* Email */}
            <div className="flex gap-2">
              <Input
                type="email"
                placeholder="Email address"
                value={emailAddress}
                onChange={(e) => setEmailAddress(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSendEmail(); }}
              />
              <Button onClick={handleSendEmail} className="gap-2 whitespace-nowrap">
                <Mail className="w-4 h-4" />Email
              </Button>
            </div>

            {/* SMS */}
            <div className="flex gap-2">
              <Input
                type="tel"
                placeholder="+91XXXXXXXXXX"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSendSMS(); }}
              />
              <Button onClick={handleSendSMS} className="gap-2 whitespace-nowrap">
                <MessageSquare className="w-4 h-4" />SMS
              </Button>
            </div>

            {/* WhatsApp */}
            <Button
              onClick={handleShareWhatsApp}
              className="w-full gap-2 bg-green-600 hover:bg-green-700 text-white"
            >
              <Send className="w-4 h-4" />
              Share via WhatsApp
            </Button>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}