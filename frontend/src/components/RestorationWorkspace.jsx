import { useState } from "react";
import { restore } from "../api";
import ImageInput from "./ImageInput";
import { Bars, Card, ErrorBox, Icon, ImagePanel, PageHeader, Spinner, StackedBar, Stat, pretty } from "./ui";

const CORRUPTIONS = [
  { id: "none", label: "None" },
  { id: "salt_pepper", label: "Salt & Pepper" },
  { id: "blur", label: "Gaussian Blur" },
  { id: "occlusion", label: "Occlusion" },
];
const SEVERITY_TEXT = {
  salt_pepper: { low: "p=0.03", medium: "p=0.08", high: "p=0.15" },
  blur: { low: "k=3 σ=0.7", medium: "k=5 σ=1.5", high: "k=7 σ=2.5" },
  occlusion: { low: "1 box 10%", medium: "2 box 20%", high: "3 box 35%" },
};
const SEV_SHORT = { low: "Low", medium: "Med", high: "High" };

function describe(c) {
  if (!c || !c.type || c.type.startsWith("none")) return "as uploaded";
  if (c.type === "salt_pepper") return `p = ${c.prob}`;
  if (c.type === "blur") return `k=${c.ksize} σ=${c.sigma}`;
  if (c.type === "occlusion") return `${c.n_rects} box · ${(c.coverage * 100).toFixed(1)}%`;
  return c.type;
}

export default function RestorationWorkspace({ mode, title, eyebrow, description, model, runLabel }) {
  const [image, setImage] = useState(null);
  const [corruption, setCorruption] = useState("blur");
  const [severity, setSeverity] = useState("medium");
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run() {
    if (!image) return setError("Choose a sample or upload an image first.");
    setLoading(true);
    setError(null);
    try {
      const t0 = performance.now();
      const r = await restore(mode, { file: image.file, sample: image.sample, corruption, severity, seed });
      r.roundtrip_ms = performance.now() - t0;
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setImage(null);
    setCorruption("blur");
    setSeverity("medium");
    setSeed(42);
    setResult(null);
    setError(null);
  }

  const m = result?.metrics;
  const corrTag = result ? (result.corruption.severity ? `${pretty(result.corruption.type)} (${SEV_SHORT[result.corruption.severity]})` : "Uploaded") : null;
  const gain = (a, b, d, unit = "") => (a != null && b != null ? `${b - a >= 0 ? "+" : ""}${(b - a).toFixed(d)}${unit} vs input` : null);

  const routingCards =
    mode === "hard" && result ? (
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Degradation classifier" subtitle="softmax probabilities p = C(x̃)" icon="analytics">
          <Bars values={result.probabilities} highlight={result.predicted} />
        </Card>
        <Card title="Router decision" subtitle="r = argmax p → exactly one specialist; clean → identity bypass" icon="alt_route">
          <div className="mb-4 flex flex-wrap gap-2">
            <span className="pill bg-primary-container/20 text-primary-fixed">Predicted: {pretty(result.predicted)}</span>
            <span className="pill bg-tertiary-container/30 text-tertiary">Expert: {result.expert}</span>
          </div>
          <div className="space-y-2.5 font-mono text-[12px]">
            <Row
              k="True corruption"
              v={
                result.true_corruption ? (
                  <span className={result.true_corruption === result.predicted ? "text-tertiary" : "text-error"}>
                    {pretty(result.true_corruption)} {result.true_corruption === result.predicted ? "✓ routed correctly" : "✗ misrouted"}
                  </span>
                ) : (
                  <span className="text-outline">unknown (uploaded)</span>
                )
              }
            />
            <Row k="Classifier latency" v={`${result.timing_ms.classifier} ms`} />
            <Row k="Expert latency" v={`${result.timing_ms.expert} ms`} />
          </div>
        </Card>
      </div>
    ) : mode === "moe" && result ? (
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Routing weights" subtitle="w = softmax(G(x̃) / τ) over 4 branches" icon="hub">
          <Bars values={result.weights} highlight={result.dominant} />
        </Card>
        <Card title="Expert contribution" subtitle="x̂ = w₀·x̃ + w₁·A_salt + w₂·A_blur + w₃·A_occ" icon="join_inner">
          <StackedBar values={result.weights} />
          <div className="mt-5 space-y-2.5 font-mono text-[12px]">
            <Row k="Strongest contributor" v={<span className="pill bg-primary-container/20 text-primary-fixed">{pretty(result.dominant)}</span>} />
            <Row k="Ranking" v={result.ranking.map(pretty).join(" › ")} />
            <Row k="Routing entropy" v={`${result.entropy} / 1.386 ${result.entropy < 0.4 ? "(sharp)" : result.entropy > 1.0 ? "(distributed)" : "(mixed)"}`} />
            <Row k="True corruption" v={result.true_corruption ? pretty(result.true_corruption) : "unknown (uploaded)"} />
          </div>
        </Card>
      </div>
    ) : null;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        eyebrow={eyebrow}
        title={title}
        description={description}
        right={
          <div className="flex items-center gap-4 self-start rounded-xl bg-surface-container-low px-4 py-2 shadow-sm md:self-auto">
            <div className="flex flex-col">
              <span className="font-mono text-[11px] uppercase text-outline">ONNX model</span>
              <span className="font-mono text-[13px] font-semibold text-on-surface">{model}</span>
            </div>
            <div className="h-6 w-px bg-surface-container-highest" />
            <span className="pill bg-surface-container-highest text-tertiary">CPU · ORT</span>
          </div>
        }
      />

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12">
        <div className="flex flex-col gap-4 lg:col-span-4">
          <Card
            title="Input & Restoration Controls"
            icon="tune"
            right={
              <button onClick={reset} className="rounded-lg p-1 text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface" title="Reset">
                <Icon name="restart_alt" className="text-lg" />
              </button>
            }
          >
            <div className="flex flex-col gap-5">
              <ImageInput kind="pets" value={image} onChange={setImage} />
              <div className="flex flex-col gap-1.5">
                <span className="font-mono text-[13px] font-medium text-on-surface-variant">Synthetic Corruption Type</span>
                <div className="grid grid-cols-2 gap-1.5">
                  {CORRUPTIONS.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setCorruption(c.id)}
                      className={`flex items-center justify-between rounded-lg px-2.5 py-1.5 text-xs transition ${
                        corruption === c.id
                          ? "bg-gradient-to-r from-primary-container to-secondary-container font-semibold text-on-primary-container shadow-sm"
                          : "bg-surface-container-high text-on-surface-variant hover:text-on-surface"
                      }`}
                    >
                      <span>{c.label}</span>
                      {corruption === c.id && <Icon name="check" className="text-base" />}
                    </button>
                  ))}
                </div>
                {corruption === "none" && <p className="text-[11px] text-outline">Use “None” when the uploaded image is already corrupted.</p>}
              </div>
              {corruption !== "none" && (
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[13px] font-medium text-on-surface-variant">Degradation Severity</span>
                    <span className="font-mono text-[11px] font-semibold text-primary">{SEVERITY_TEXT[corruption][severity]}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 rounded-xl bg-surface-container-lowest p-1">
                    {["low", "medium", "high"].map((s) => (
                      <button
                        key={s}
                        onClick={() => setSeverity(s)}
                        className={`rounded-lg px-1 py-1 text-center font-mono text-[12px] transition ${
                          severity === s
                            ? "bg-surface-container-high font-semibold text-primary-fixed shadow-[0_0_12px_rgba(192,193,255,0.15)]"
                            : "text-on-surface-variant hover:text-on-surface"
                        }`}
                      >
                        {SEV_SHORT[s]}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-end gap-1 font-mono text-[11px] text-outline">
                    random seed
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      className="w-20 rounded bg-surface-container-lowest px-1.5 py-0.5 text-on-surface-variant"
                    />
                    <button onClick={() => setSeed(Math.floor(Math.random() * 1e6))} className="text-primary" title="Random seed">
                      <Icon name="casino" className="text-base" />
                    </button>
                  </div>
                </div>
              )}
              <button onClick={run} disabled={loading} className="btn-primary">
                {loading ? <Spinner /> : <Icon name="auto_fix_high" className="text-xl" />}
                <span>{loading ? "Running…" : runLabel}</span>
              </button>
              <ErrorBox error={error} />
            </div>
          </Card>
          <div className="flex flex-col gap-1.5 rounded-xl bg-surface-container-low/70 p-4 font-mono text-[12px]">
            <div className="flex justify-between text-outline">
              <span>DISPATCH STATUS</span>
              <span className="text-tertiary">{loading ? "RUNNING" : result ? "DONE // READY" : "IDLE // READY"}</span>
            </div>
            <div className="flex justify-between text-on-surface-variant">
              <span>Input tensor</span>
              <span className="text-on-surface">1 × 3 × 128 × 128</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6 lg:col-span-8">
          {routingCards}
          <div className={`grid grid-cols-1 gap-4 sm:grid-cols-2 ${result?.error_map ? "xl:grid-cols-4" : "md:grid-cols-3"}`}>
            {(result?.original || !result) && <ImagePanel label="Clean / Original" src={result?.original} filename="original.png" tag="Reference" footer="ground truth" />}
            <ImagePanel label="Corrupted Input" src={result?.input} filename="input.png" tag={corrTag} tagTone="error" caption={result && describe(result.corruption)} footer="model input" />
            <ImagePanel label="Restored Output" src={result?.output} filename={`restored_${mode}.png`} tag={result ? "Restored" : null} tagTone="tertiary" footer="128 × 128 px" highlight />
            {result?.error_map && <ImagePanel label="|Error| Map" src={result.error_map} filename="error_map.png" tag="vs clean" footer="mean abs. error" />}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Inference time" icon="bolt" iconColor="text-tertiary" value={result ? result.inference_ms.toFixed(1) : null} unit="ms" hint={result && `round-trip ${result.roundtrip_ms.toFixed(0)} ms`} />
            <Stat label="Peak SNR" icon="show_chart" accent="text-tertiary" value={m ? (m.psnr_output ?? "∞") : null} unit="dB" hint={m && gain(m.psnr_input, m.psnr_output, 2, " dB")} />
            <Stat label="Structural sim." icon="layers" iconColor="text-primary-container" accent="text-primary" value={m ? m.ssim_output : null} hint={m && gain(m.ssim_input, m.ssim_output, 3)} />
            <Stat label="Active pipeline" icon="schema" iconColor="text-outline" value={result ? pretty(result.corruption.type?.split(" ")[0]) : null} hint={result ? describe(result.corruption) : null} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-outline-variant/30 pb-2 last:border-0">
      <span className="text-outline">{k}</span>
      <span className="text-right text-on-surface">{v}</span>
    </div>
  );
}
