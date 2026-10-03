import { useEffect, useRef, useState } from "react";
import { getSamples, sampleUrl } from "../api";

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
        <div className="overflow-hidden rounded-xl border border-indigo-500/40">
          <video ref={videoRef} autoPlay playsInline className="aspect-video w-full bg-black object-cover" />
          <div className="flex gap-2 p-2">
            <button onClick={capture} className="flex-1 rounded-lg bg-indigo-500 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-400">
              Capture photo
            </button>
            <button onClick={stopCam} className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800">
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
          className={`flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed p-3 transition ${
            drag ? "border-indigo-400 bg-indigo-500/10" : "border-slate-700 hover:border-slate-500"
          }`}
        >
          {value?.preview ? (
            <img src={value.preview} alt="selected" className="h-16 w-16 rounded-lg object-cover" />
          ) : (
            <div className="flex h-16 w-16 items-center justify-center rounded-lg bg-slate-800 text-2xl text-slate-500">⇪</div>
          )}
          <div className="min-w-0 text-sm">
            <div className="font-medium text-slate-200">{value ? "Change image" : "Upload an image"}</div>
            <div className="truncate text-xs text-slate-500">{value?.label || "Click or drop PNG / JPEG / WEBP (max 10 MB)"}</div>
          </div>
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
        <button onClick={startCam} className="w-full rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800">
          ◉ Use webcam
        </button>
      )}
      {camError && <p className="text-xs text-rose-300">{camError}</p>}
      {samples.length > 0 && (
        <div>
          <div className="mb-1.5 text-[11px] uppercase tracking-wide text-slate-500">Or pick a sample (unseen test images)</div>
          <div className="grid grid-cols-5 gap-1.5">
            {samples.map((s) => (
              <button
                key={s}
                onClick={() => onChange({ sample: s, preview: sampleUrl(s), label: s })}
                className={`overflow-hidden rounded-lg border-2 transition ${
                  value?.sample === s ? "border-indigo-400" : "border-transparent opacity-70 hover:opacity-100"
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
