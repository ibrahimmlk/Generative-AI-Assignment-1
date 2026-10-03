import { useEffect, useRef, useState } from "react";
import { getSamples, sampleUrl } from "../api";
import { Icon } from "./ui";

// Upload / drag-and-drop / sample picker / optional webcam capture.
// onChange receives { file } or { sample } plus a preview URL.
export default function ImageInput({ kind = "pets", value, onChange, webcam = false }) {
  const [samples, setSamples] = useState([]);
  const [drag, setDrag] = useState(false);
  const [camOn, setCamOn] = useState(false);
  const [camError, setCamError] = useState(null);
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    getSamples()
      .then((s) => setSamples(s[kind] || []))
      .catch(() => setSamples([]));
  }, [kind]);

  useEffect(() => () => stopCam(), []);

  function pickFile(file) {
    if (!file) return;
    onChange({ file, preview: URL.createObjectURL(file), label: file.name });
  }

  async function startCam() {
    setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      setCamOn(true);
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      }, 0);
    } catch (e) {
      setCamError("Webcam unavailable: " + e.message);
    }
  }

  function stopCam() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCamOn(false);
  }

  function capture() {
    const v = videoRef.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => {
      const file = new File([blob], "webcam.png", { type: "image/png" });
      pickFile(file);
      stopCam();
    }, "image/png");
  }

  return (
    <div className="space-y-3">
      {camOn ? (
        <div className="overflow-hidden rounded-xl ring-1 ring-primary/40">
          <video ref={videoRef} autoPlay playsInline className="aspect-video w-full bg-black object-cover" />
          <div className="flex gap-2 p-2">
            <button onClick={capture} className="flex-1 rounded-lg bg-primary-container px-3 py-2 text-sm font-medium text-on-primary-container hover:opacity-90">
              Capture photo
            </button>
            <button onClick={stopCam} className="rounded-lg border border-outline-variant px-3 py-2 text-sm text-on-surface-variant hover:bg-surface-container-high">
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            pickFile(e.dataTransfer.files?.[0]);
          }}
          className={`group flex cursor-pointer flex-col items-center justify-center rounded-xl p-5 text-center transition-all ${
            drag ? "bg-primary-container/20 ring-2 ring-primary" : "bg-surface-container-lowest/60 hover:bg-surface-container-lowest"
          }`}
        >
          {value?.preview ? (
            <img src={value.preview} alt="selected" className="mb-2 h-20 w-20 rounded-lg object-cover shadow-md" />
          ) : (
            <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-surface-container text-primary shadow-sm transition-transform group-hover:scale-110">
              <Icon name="cloud_upload" className="text-2xl" />
            </div>
          )}
          <span className="text-sm font-medium text-on-surface">{value ? "Change image" : "Drag & drop image here or browse files"}</span>
          <span className="mt-1 max-w-full truncate font-mono text-[11px] font-semibold uppercase tracking-wider text-outline">
            {value?.label || "PNG, JPG, WebP up to 10MB"}
          </span>
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            className="hidden"
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
        </div>
      )}
      {webcam && !camOn && (
        <button onClick={startCam} className="flex w-full items-center justify-center gap-2 rounded-lg bg-surface-container-high px-3 py-2 text-sm text-on-surface-variant hover:text-on-surface">
          <Icon name="photo_camera" className="text-lg" /> Use webcam
        </button>
      )}
      {camError && <p className="text-xs text-error">{camError}</p>}
      {samples.length > 0 && (
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <span className="font-mono text-[13px] font-medium text-on-surface-variant">Sample Test Harness</span>
            <span className="font-mono text-[11px] font-semibold text-outline">{samples.length} UNSEEN</span>
          </div>
          <div className="grid grid-cols-5 gap-1.5">
            {samples.map((s) => (
              <button
                key={s}
                onClick={() => onChange({ sample: s, preview: sampleUrl(s), label: s })}
                className={`overflow-hidden rounded-lg transition-transform hover:scale-105 ${
                  value?.sample === s ? "shadow-md ring-2 ring-primary" : "opacity-70 hover:opacity-100"
                }`}
              >
                <img src={sampleUrl(s)} alt={s} className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
