import { useState } from "react";
import { sketch } from "../api";
import ImageInput from "./ImageInput";
import { Card, ErrorBox, Icon, ImagePanel, PageHeader, Spinner, Stat, download } from "./ui";

const STYLES = [
  { id: 1, name: "Style 1", hint: "FS2K style category 1", icon: "edit" },
  { id: 2, name: "Style 2", hint: "FS2K style category 2", icon: "brush" },
  { id: 3, name: "Style 3", hint: "FS2K style category 3", icon: "polyline" },
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
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow="TASK 4 // STYLE-CONDITIONED PIX2PIX GAN"
        title="Face-to-Sketch Generator"
        description="U-Net generator G(x, s) with a learned style embedding, trained adversarially against a style-conditioned PatchGAN discriminator on paired FS2K photographs and sketches."
        right={
          <div className="flex items-center gap-4 self-start rounded-xl bg-surface-container-low px-4 py-2 shadow-sm md:self-auto">
            <div className="flex flex-col">
              <span className="font-mono text-[11px] uppercase text-outline">ONNX model</span>
              <span className="font-mono text-[13px] font-semibold text-on-surface">sketch_generator.onnx</span>
            </div>
            <div className="h-6 w-px bg-surface-container-highest" />
            <span className="pill bg-surface-container-highest text-tertiary">Generator only</span>
          </div>
        }
      />
      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <Card title="Portrait Input & Style" icon="tune" subtitle="Faces are centre-cropped to a square and resized to 128×128">
            <div className="flex flex-col gap-5">
              <ImageInput kind="faces" value={image} onChange={setImage} webcam />
              <div className="flex flex-col gap-1.5">
                <span className="font-mono text-[13px] font-medium text-on-surface-variant">Sketch Style Condition</span>
                <div className="grid grid-cols-3 gap-2">
                  {STYLES.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => setStyle(s.id)}
                      className={`flex flex-col items-start gap-1 rounded-xl p-3 text-left transition ${
                        style === s.id
                          ? "bg-gradient-to-br from-primary-container/40 to-secondary-container/40 ring-2 ring-primary"
                          : "bg-surface-container-high hover:bg-surface-container-highest"
                      }`}
                    >
                      <div className="flex w-full items-center justify-between">
                        <Icon name={s.icon} className={`text-lg ${style === s.id ? "text-primary-fixed" : "text-outline"}`} />
                        <Icon name={style === s.id ? "radio_button_checked" : "radio_button_unchecked"} className="text-base text-primary" />
                      </div>
                      <div className={`font-display text-sm font-semibold ${style === s.id ? "text-primary-fixed" : "text-on-surface"}`}>{s.name}</div>
                      <div className="text-[10px] leading-tight text-outline">{s.hint}</div>
                    </button>
                  ))}
                </div>
              </div>
              <button onClick={run} disabled={loading} className="btn-primary">
                {loading ? <Spinner /> : <Icon name="draw" className="text-xl" />}
                <span>{loading ? "Generating…" : "Generate Sketch"}</span>
              </button>
              <ErrorBox error={error} />
            </div>
          </Card>
        </div>
        <div className="flex flex-col gap-6 lg:col-span-8">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <ImagePanel label="Original Photo" src={result?.photo} filename="photo.png" tag="Input" footer="128 × 128 RGB" />
            <ImagePanel
              label="Generated Sketch"
              src={result?.sketch}
              filename={`sketch_style${result?.style}.png`}
              tag={result ? `Style ${result.style}` : null}
              tagTone="tertiary"
              footer="G(x, s) output"
              highlight
            />
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Inference time" icon="timer" iconColor="text-tertiary" value={result ? result.inference_ms.toFixed(1) : null} unit="ms" hint={result && `round-trip ${result.roundtrip_ms.toFixed(0)} ms`} />
            <Stat label="Style condition" icon="linear_scale" value={result ? `Style ${result.style}` : null} hint="learned embedding" />
            <Stat label="Resolution" icon="crop" value="128²" hint="generator in / out" />
            <div className="flex flex-col justify-center gap-2 rounded-xl bg-surface-container-low/80 p-4 shadow-sm">
              <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-outline">Export</span>
              <button
                disabled={!result}
                onClick={() => download(result.sketch, `sketch_style${result.style}.png`)}
                className="flex items-center justify-center gap-2 rounded-lg bg-primary-container px-3 py-2 text-sm font-semibold text-on-primary-container hover:opacity-90 disabled:opacity-40"
              >
                <Icon name="download" className="text-lg" /> Download sketch
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
