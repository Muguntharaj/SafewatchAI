/**
 * AddCamera.tsx — FIXES:
 *  1. FIX 4 (vertical layout): card is now full-width, not max-w-xl centered.
 *     Camera type buttons stack vertically on narrow screens.
 *  2. FIX 4 (notify parent): onAdded callback so Dashboard adds the new camera
 *     to the live grid instantly without a page refresh.
 *  3. FIX (backend inputs): Added WiFi as a valid camera_type option.
 *     stream_url placeholder adapts per type.
 */

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Camera, Plus, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { CameraInfo } from "./CameraGrid";

interface CameraFormData {
  camera_id: string;
  name: string;
  location: string;
  stream_url: string;
  camera_type: "IP" | "WiFi" | "USB";
  fps: number;
  resolution: string;
}

type SubmitStatus = "idle" | "loading" | "success" | "error";

interface AddCameraProps {
  /** Called after a camera is successfully added so Dashboard can update the grid */
  onAdded?: (cam: CameraInfo) => void;
}

export function AddCamera({ onAdded }: AddCameraProps) {
  const [camera, setCamera] = useState<CameraFormData>({
    camera_id:   "",
    name:        "",
    location:    "",
    stream_url:  "",
    camera_type: "IP",
    fps:         25,
    resolution:  "1920x1080",
  });

  const [status, setStatus]           = useState<SubmitStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [addedCamera, setAddedCamera]   = useState<string>("");

  const handleChange = (field: keyof CameraFormData, value: string | number) => {
    setCamera((prev) => ({ ...prev, [field]: value }));
  };

  const streamUrlPlaceholder = () => {
    if (camera.camera_type === "USB")  return "0  (USB device index, e.g. 0 or 1)";
    if (camera.camera_type === "WiFi") return "http://192.168.1.x:8080/video";
    return "rtsp://user:pass@192.168.1.x:554/stream";
  };

  const streamUrlHint = () => {
    if (camera.camera_type === "USB")  return "USB device index: 0 for first USB camera, 1 for second, etc.";
    if (camera.camera_type === "WiFi") return "HTTP/MJPEG stream URL from the WiFi camera app";
    return "Full RTSP URL — rtsp://user:pass@host:554/stream";
  };

  const handleSubmit = async () => {
    // ── Validation ────────────────────────────────────────────────────────
    if (!camera.camera_id.trim()) {
      setStatus("error"); setErrorMessage("Camera ID is required."); return;
    }
    if (!camera.name.trim()) {
      setStatus("error"); setErrorMessage("Camera name is required."); return;
    }
    if (!camera.stream_url.trim()) {
      setStatus("error"); setErrorMessage("Stream URL is required."); return;
    }

    setStatus("loading");
    setErrorMessage("");

    const payload = {
      camera_id:   camera.camera_id.trim(),
      name:        camera.name.trim(),
      location:    camera.location.trim(),
      stream_url:  camera.stream_url.trim(),
      camera_type: camera.camera_type,
      fps:         camera.fps,
      resolution:  camera.resolution,
    };

    try {
      const res = await fetch("http://localhost:8000/api/cameras/add", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail ?? `Server error: ${res.status}`);
      }

      const created = await res.json();
      setAddedCamera(camera.name || camera.camera_id);
      setStatus("success");

      // Notify parent so camera appears in grid immediately
      if (onAdded) {
        onAdded({
          id:       created.camera_id ?? payload.camera_id,
          name:     created.name      ?? payload.name,
          location: created.location  ?? payload.location,
          status:   'active',
        });
      }

      // Reset form
      setCamera({
        camera_id:   "",
        name:        "",
        location:    "",
        stream_url:  "",
        camera_type: "IP",
        fps:         25,
        resolution:  "1920x1080",
      });
    } catch (err: any) {
      setStatus("error");
      setErrorMessage(err.message ?? "Failed to add camera. Please try again.");
    }
  };

  return (
    // FIX 4: full-width layout, no max-w-xl restriction
    <div className="w-full p-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="shadow-xl border-2 border-blue-200">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg">
            <CardTitle className="flex items-center gap-2 text-xl">
              <Camera className="w-6 h-6" />
              Add New Camera
            </CardTitle>
            <CardDescription className="text-blue-100">
              Connect a new RTSP / IP, WiFi or USB camera to SafeWatch AI
            </CardDescription>
          </CardHeader>

          {/* FIX 4: Two-column layout on md+, single column on mobile */}
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">

              {/* ── Left column ── */}
              <div className="space-y-5">
                {/* Success Banner */}
                {status === "success" && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-3 p-4 bg-green-50 border border-green-300 rounded-lg text-green-800 md:col-span-2"
                  >
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <div>
                      <p className="font-semibold">Camera added successfully!</p>
                      <p className="text-sm text-green-700">
                        <strong>{addedCamera}</strong> is now being monitored.
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* Error Banner */}
                {status === "error" && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-3 p-4 bg-red-50 border border-red-300 rounded-lg text-red-800"
                  >
                    <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                    <div>
                      <p className="font-semibold">Failed to add camera</p>
                      <p className="text-sm text-red-700">{errorMessage}</p>
                    </div>
                  </motion.div>
                )}

                {/* Camera Type — vertical stack */}
                <div className="space-y-2">
                  <Label className="text-base font-semibold">Camera Type</Label>
                  {/* FIX 4: vertical list */}
                  <div className="flex flex-col gap-2">
                    {(["IP", "WiFi", "USB"] as const).map((type) => (
                      <button
                        key={type}
                        type="button"
                        onClick={() => handleChange("camera_type", type)}
                        className={`w-full py-3 px-4 rounded-lg border-2 font-semibold text-sm text-left transition-all flex items-center gap-3 ${
                          camera.camera_type === type
                            ? "border-blue-600 bg-blue-50 text-blue-700"
                            : "border-gray-200 text-gray-600 hover:border-blue-300"
                        }`}
                      >
                        <span className="text-lg">
                          {type === "IP" ? "📡" : type === "WiFi" ? "📶" : "🔌"}
                        </span>
                        <span>
                          {type === "IP"   ? "RTSP / IP Camera"
                          : type === "WiFi" ? "WiFi / HTTP Camera"
                          :                   "USB Camera"}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Camera ID */}
                <div className="space-y-2">
                  <Label htmlFor="camera_id" className="text-base font-semibold">
                    Camera ID <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    id="camera_id"
                    placeholder="e.g. cam-4"
                    value={camera.camera_id}
                    onChange={(e) => handleChange("camera_id", e.target.value)}
                    className="h-11"
                  />
                  <p className="text-xs text-gray-500">
                    Must be unique. Use format: cam-1, cam-2, cam-3…
                  </p>
                </div>

                {/* Camera Name */}
                <div className="space-y-2">
                  <Label htmlFor="name" className="text-base font-semibold">
                    Camera Name <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    id="name"
                    placeholder="e.g. Front Gate Camera"
                    value={camera.name}
                    onChange={(e) => handleChange("name", e.target.value)}
                    className="h-11"
                  />
                </div>
              </div>

              {/* ── Right column ── */}
              <div className="space-y-5">
                {/* Location */}
                <div className="space-y-2">
                  <Label htmlFor="location" className="text-base font-semibold">
                    Location
                  </Label>
                  <Input
                    id="location"
                    placeholder="e.g. Main Entrance, Parking Lot"
                    value={camera.location}
                    onChange={(e) => handleChange("location", e.target.value)}
                    className="h-11"
                  />
                </div>

                {/* Stream URL */}
                <div className="space-y-2">
                  <Label htmlFor="stream_url" className="text-base font-semibold">
                    Stream URL <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    id="stream_url"
                    placeholder={streamUrlPlaceholder()}
                    value={camera.stream_url}
                    onChange={(e) => handleChange("stream_url", e.target.value)}
                    className="h-11 font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500">{streamUrlHint()}</p>
                </div>

                {/* FPS + Resolution */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="fps" className="text-base font-semibold">FPS</Label>
                    <Input
                      id="fps"
                      type="number"
                      min={1}
                      max={60}
                      value={camera.fps}
                      onChange={(e) => handleChange("fps", parseInt(e.target.value) || 25)}
                      className="h-11"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="resolution" className="text-base font-semibold">
                      Resolution
                    </Label>
                    <select
                      id="resolution"
                      value={camera.resolution}
                      onChange={(e) => handleChange("resolution", e.target.value)}
                      className="w-full h-11 rounded-md border border-gray-300 px-3 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="3840x2160">4K (3840×2160)</option>
                      <option value="1920x1080">Full HD (1920×1080)</option>
                      <option value="1280x720">HD (1280×720)</option>
                      <option value="640x480">SD (640×480)</option>
                    </select>
                  </div>
                </div>

                {/* Submit */}
                <Button
                  type="button"
                  onClick={handleSubmit}
                  disabled={status === "loading"}
                  className="w-full h-12 text-base bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 mt-2"
                >
                  {status === "loading" ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Adding Camera…
                    </>
                  ) : (
                    <>
                      <Plus className="w-5 h-5 mr-2" />
                      Add Camera
                    </>
                  )}
                </Button>

                <p className="text-xs text-center text-gray-500">
                  The camera will start streaming automatically after being added.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
