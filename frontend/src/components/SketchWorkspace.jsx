import { useState } from "react";
import { sketch } from "../api";
import ImageInput from "./ImageInput";
import { Card, ErrorBox, ImagePanel, Spinner, Stat, download } from "./ui";

const STYLES = [
  { id: 1, name: "Style 1", hint: "FS2K style category 1" },
  { id: 2, name: "Style 2", hint: "FS2K style category 2" },
  { id: 3, name: "Style 3", hint: "FS2K style category 3" },
];

export default function SketchWorkspace() {
  const [image, setImage] = useState(null);
  const [style, setStyle] = useState(1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    if (!image) return setError("Upload a face photo, capture one with the webcam, or pick a sample.");
    setLoading(true);
    setError(null);
    try {
      const t0 = performance.now();
      const r = await sketch({ file: image.file, sample: image.sample, style });
      r.roundtrip_ms = performance.now() - t0;
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">Face-to-Sketch Generator</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Conditional pix2pix GAN (U-Net generator + PatchGAN discriminator) trained on FS2K. The selected style is fed to the
          generator as a learned categorical embedding.
        </p>
      </div>
      <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
        <Card title="Input" subtitle="Faces are centre-cropped to a square and resized to 128×128">
          <div className="space-y-5">
            <ImageInput kind="faces" value={image} onChange={setImage} webcam />
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-wide text-slate-500">Sketch style</div>
              <div className="grid grid-cols-3 gap-2">
                {STYLES.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setStyle(s.id)}
                    className={`rounded-xl border p-3 text-left transition ${
                      style === s.id ? "border-indigo-400 bg-indigo-500/15" : "border-slate-700 hover:border-slate-500"
                    }`}
                  >
                    <div className={`text-sm font-semibold ${style === s.id ? "text-indigo-200" : "text-slate-200"}`}>{s.name}</div>
                    <div className="mt-0.5 text-[10px] leading-tight text-slate-500">{s.hint}</div>
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={run}
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:opacity-60"
            >
              {loading && <Spinner />} Generate sketch
            </button>
            <ErrorBox error={error} />
          </div>
        </Card>
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <ImagePanel label="Original photograph" src={result?.photo} filename="photo.png" />
            <ImagePanel label={`Generated sketch${result ? ` — Style ${result.style}` : ""}`} src={result?.sketch} filename={`sketch_style${result?.style}.png`} />
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Inference time" value={result ? `${result.inference_ms.toFixed(1)} ms` : null} hint={result && `round-trip ${result.roundtrip_ms.toFixed(0)} ms`} accent="text-indigo-300" />
            <Stat label="Style condition" value={result ? `Style ${result.style}` : null} hint="embedding index" />
            <Stat label="Resolution" value="128 × 128" hint="generator input / output" />
            <div className="flex items-center rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-3">
              <button
                disabled={!result}
                onClick={() => download(result.sketch, `sketch_style${result.style}.png`)}
                className="w-full rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-slate-100 hover:bg-slate-700 disabled:opacity-40"
              >
                ↓ Download sketch
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
