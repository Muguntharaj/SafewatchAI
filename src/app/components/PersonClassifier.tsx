/**
 * PersonClassifier.tsx — Final Fixed Build
 *
 * VIDEO BUGS FIXED:
 *  1. "Unknown visitor" info banner still showed even after switching category
 *     to Employee/Owner — now hidden as soon as category changes away from Unknown.
 *  2. Category dropdown stayed "Unknown" visually even after user selected
 *     Employee — fixed by using controlled state with immediate re-render.
 *  3. After Save & Classify the dialog auto-closes after 1.2s with green success.
 *  4. Image loads correctly for unknown persons (full http:// URL from backend).
 *  5. Fallback URL chain: unknown → known folder swap.
 *  6. justSaved state disables all inputs to prevent double-submit.
 */

import { useState, useEffect } from 'react';
import { Button }   from './ui/button';
import {
  Dialog, DialogContent, DialogDescription,
  DialogFooter, DialogHeader, DialogTitle,
} from './ui/dialog';
import { Label }  from './ui/label';
import { Input }  from './ui/input';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from './ui/select';
import { User, UserCheck, UserX, Download, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';

const API = 'http://localhost:8000';

export interface Person {
  id:            string;
  name?:         string;
  category:      'Employee' | 'Owner' | 'Unknown';
  confidence:    number;
  timestamp:     Date;
  imageData?:    string;
  cameraId?:     string;
  recognizedAs?: string;
}

interface PersonClassifierProps {
  person:      Person;
  isOpen:      boolean;
  onClose:     () => void;
  onClassify:  (id: string, name: string, category: 'Employee' | 'Owner' | 'Unknown') => void;
  onDownload:  (person: Person) => void;
}

function buildFaceUrls(imageData?: string): string[] {
  if (!imageData) return [];
  if (imageData.startsWith('http://') || imageData.startsWith('https://')) {
    const alt = imageData.includes('/unknown/')
      ? imageData.replace('/unknown/', '/known/')
      : imageData.replace('/known/', '/unknown/');
    return imageData !== alt ? [imageData, alt] : [imageData];
  }
  const parts    = imageData.split(/[/\\]/);
  const filename = parts[parts.length - 1];
  if (!filename || filename.includes(':') || !filename.includes('.')) return [];
  return [
    `${API}/media/faces/unknown/${filename}`,
    `${API}/media/faces/known/${filename}`,
  ];
}

export function PersonClassifier({
  person, isOpen, onClose, onClassify, onDownload,
}: PersonClassifierProps) {
  const [name,      setName]      = useState(person.name || '');
  const [category,  setCategory]  = useState<'Employee' | 'Owner' | 'Unknown'>(person.category ?? 'Unknown');
  const [urlIndex,  setUrlIndex]  = useState(0);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgFailed, setImgFailed] = useState(false);
  const [justSaved, setJustSaved] = useState(false);

  const urls       = buildFaceUrls(person.imageData);
  const currentUrl = urls[urlIndex] ?? null;

  // Re-sync local state whenever dialog opens for a different person
  useEffect(() => {
    setUrlIndex(0);
    setImgLoaded(false);
    setImgFailed(urls.length === 0);
    setName(person.name || '');
    setCategory(person.category ?? 'Unknown');
    setJustSaved(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [person.id, person.imageData, person.category, person.name, isOpen]);

  const handleImgError = () => {
    if (urlIndex + 1 < urls.length) {
      setUrlIndex(i => i + 1);
      setImgLoaded(false);
    } else {
      setImgFailed(true);
    }
  };

  const handleSave = () => {
    if (!name.trim() || justSaved) return;
    onClassify(person.id, name.trim(), category);
    setJustSaved(true);
    setTimeout(onClose, 1200);
  };

  const handleDownload = async () => {
    if (!currentUrl || imgFailed) { onDownload(person); return; }
    try {
      const res  = await fetch(currentUrl);
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const a    = document.createElement('a');
      a.href     = URL.createObjectURL(blob);
      a.download = `person_${person.id}.jpg`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
      toast.success('Image downloaded');
    } catch { onDownload(person); }
  };

  const showPlaceholder = !currentUrl || imgFailed;

  // FIX: badge color tracks the current SELECTED category, not person.category
  const categoryColor =
    category === 'Employee' ? 'bg-blue-100 text-blue-800 border-blue-200' :
    category === 'Owner'    ? 'bg-purple-100 text-purple-800 border-purple-200' :
                              'bg-orange-100 text-orange-800 border-orange-200';

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[580px] max-h-[92vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {justSaved
              ? <CheckCircle className="w-5 h-5 text-green-600" />
              : person.category === 'Unknown'
              ? <UserX     className="w-5 h-5 text-orange-600" />
              : <UserCheck className="w-5 h-5 text-green-600" />}
            {justSaved
              ? 'Classified Successfully!'
              : person.recognizedAs ? 'Identity Confirmed' : 'Person Classification'}
          </DialogTitle>
          <DialogDescription>
            Detected at {person.timestamp.toLocaleString()} with{' '}
            {(person.confidence * 100).toFixed(0)}% confidence
            {person.recognizedAs && !justSaved && (
              <span className="block mt-1 text-green-600">
                ✓ Auto-recognized as: {person.recognizedAs}
              </span>
            )}
            {justSaved && (
              <span className="block mt-1 text-green-600 font-medium">
                ✓ {name} saved as {category} — will be auto-recognized next time
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Face image */}
          <div
            className="w-full rounded-xl overflow-hidden bg-gradient-to-br from-slate-800 to-slate-900
                       border border-slate-700 flex items-center justify-center relative"
            style={{ minHeight: '300px' }}
          >
            {currentUrl && !imgLoaded && !imgFailed && (
              <div className="absolute inset-0 bg-slate-700 animate-pulse" />
            )}
            {currentUrl && !imgFailed && (
              <img
                key={currentUrl}
                src={currentUrl}
                alt="Detected person"
                className={`w-full transition-opacity duration-300 ${imgLoaded ? 'opacity-100' : 'opacity-0'}`}
                style={{ maxHeight: '360px', objectFit: 'contain' }}
                onLoad={()  => setImgLoaded(true)}
                onError={handleImgError}
              />
            )}
            {showPlaceholder && (
              <div className="flex flex-col items-center gap-3 text-slate-400 py-12">
                <User className="w-28 h-28 opacity-30" />
                <span className="text-sm font-medium">No image available</span>
                {person.imageData && (
                  <span className="text-xs text-slate-500 max-w-xs text-center break-all">
                    Image could not be loaded from server
                  </span>
                )}
              </div>
            )}
            {imgLoaded && (
              <div className="absolute top-2 right-2 bg-black/60 rounded px-2 py-0.5">
                <span className="text-[10px] text-green-400 font-medium">✓ Image loaded</span>
              </div>
            )}
            {/* FIX: badge shows CURRENT selected category, updates live */}
            <div className={`absolute bottom-2 left-2 px-2 py-0.5 rounded-full text-xs font-semibold border ${categoryColor}`}>
              {category}
            </div>
          </div>

          {/* Name */}
          <div className="space-y-2">
            <Label htmlFor="pc-name">Name</Label>
            <Input
              id="pc-name"
              placeholder="Enter person's name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && name.trim()) handleSave(); }}
              disabled={justSaved}
            />
          </div>

          {/* Category — FIX: controlled value updates image badge live */}
          <div className="space-y-2">
            <Label htmlFor="pc-category">Category</Label>
            <Select
              value={category}
              onValueChange={(v) => setCategory(v as 'Employee' | 'Owner' | 'Unknown')}
              disabled={justSaved}
            >
              <SelectTrigger id="pc-category"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="Employee">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
                    Employee
                  </span>
                </SelectItem>
                <SelectItem value="Owner">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-purple-500 inline-block" />
                    Owner
                  </span>
                </SelectItem>
                <SelectItem value="Unknown">
                  <span className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-orange-500 inline-block" />
                    Unknown
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/*
            FIX: "Unknown visitor" banner only shows when category IS still Unknown
            and the person hasn't been saved yet. Disappears immediately when user
            selects Employee or Owner.
          */}
          {category === 'Unknown' && !person.recognizedAs && !justSaved && (
            <div className="flex items-center gap-2 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <UserX className="w-4 h-4 text-orange-600 shrink-0" />
              <span className="text-sm text-orange-800">
                Unknown visitor — classify to enable automatic recognition next time.
                Their face will be auto-clustered with similar unknowns.
              </span>
            </div>
          )}

          {/* Success banner */}
          {justSaved && (
            <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-lg">
              <CheckCircle className="w-4 h-4 text-green-600 shrink-0" />
              <span className="text-sm text-green-800 font-medium">
                ✅ {name} classified as <strong>{category}</strong> — saved to database
              </span>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 pt-2">
          <Button variant="outline" onClick={handleDownload} disabled={showPlaceholder || justSaved} className="gap-2">
            <Download className="w-4 h-4" />Save Image
          </Button>
          <Button variant="outline" onClick={onClose} disabled={justSaved}>Cancel</Button>
          <Button
            onClick={handleSave}
            disabled={!name.trim() || justSaved}
            className={justSaved ? 'bg-green-600 hover:bg-green-600' : ''}
          >
            {justSaved
              ? <><CheckCircle className="w-4 h-4 mr-2" />Saved!</>
              : 'Save & Classify'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}