import { useState } from "react";
import { restore } from "../api";
import ImageInput from "./ImageInput";
import { Bars, Card, ErrorBox, ImagePanel, Spinner, StackedBar, Stat, pretty } from "./ui";

const CORRUPTIONS = [
  { id: "none", label: "None" },
  { id: "salt_pepper", label: "Salt & Pepper" },
  { id: "blur", label: "Gaussian Blur" },
  { id: "occlusion", label: "Occlusion" },
];
const SEVERITY_TEXT = {
  salt_pepper: { low: "p = 0.03", medium: "p = 0.08", high: "p = 0.15" },
  blur: { low: "k=3, σ=0.7", medium: "k=5, σ=1.5", high: "k=7, σ=2.5" },
  occlusion: { low: "1 box, 10%", medium: "2 boxes, 20%", high: "3 boxes, 35%" },
};

function describe(c) {
  if (!c || !c.type || c.type.startsWith("none")) return "None — image used as uploaded";
  if (c.type === "salt_pepper") return `Salt & pepper, p = ${c.prob}`;
  if (c.type === "blur") return `Blur, kernel ${c.ksize}, σ = ${c.sigma}`;
  if (c.type === "occlusion") return `${c.n_rects} box(es), ${(c.coverage * 100).toFixed(1)}% area`;
  return c.type;
}

export default function RestorationWorkspace({ mode, title, description }) {
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
      const r = await restore(mode, {
        file: image.file,
        sample: image.sample,
        corruption,
        severity,
        seed,
      });
      r.roundtrip_ms = performance.now() - t0;
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const m = result?.metrics;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">{title}</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">{description}</p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[340px_1fr]">
        <Card title="Input" subtitle="Upload a corrupted image, or pick a clean one and corrupt it at runtime">
          <div className="space-y-5">
            <ImageInput kind="pets" value={image} onChange={setImage} />
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-wide text-slate-500">Runtime corruption</div>
              <div className="grid grid-cols-2 gap-2">
                {CORRUPTIONS.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setCorruption(c.id)}
                    className={`rounded-lg border px-3 py-2 text-xs font-medium transition ${
                      corruption === c.id
                        ? "border-indigo-400 bg-indigo-500/15 text-indigo-200"
                        : "border-slate-700 text-slate-300 hover:border-slate-500"
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              {corruption === "none" && (
                <p className="mt-2 text-[11px] text-slate-500">Use this when the uploaded image is already corrupted.</p>
              )}
            </div>
            {corruption !== "none" && (
              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wide text-slate-500">Severity</div>
                <div className="grid grid-cols-3 rounded-lg bg-slate-800 p-1">
                  {["low", "medium", "high"].map((s) => (
                    <button
                      key={s}
                      onClick={() => setSeverity(s)}
                      className={`rounded-md py-1.5 text-xs capitalize transition ${
                        severity === s ? "bg-indigo-500 text-white shadow" : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                  <span>{SEVERITY_TEXT[corruption][severity]}</span>
                  <span className="flex items-center gap-1">
                    seed
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      className="w-16 rounded border border-slate-700 bg-slate-950 px-1.5 py-0.5 text-slate-300"
                    />
                    <button onClick={() => setSeed(Math.floor(Math.random() * 1e6))} className="px-1 text-indigo-300" title="Random seed">
                      ⟳
                    </button>
                  </span>
                </div>
              </div>
            )}
            <button
              onClick={run}
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:opacity-60"
            >
              {loading && <Spinner />} Restore image
            </button>
            <ErrorBox error={error} />
          </div>
        </Card>

        <div className="space-y-6">
          <div className={`grid gap-4 sm:grid-cols-2 ${result?.original ? "lg:grid-cols-4" : "lg:grid-cols-3"}`}>
            {(result?.original || !result) && <ImagePanel label="Clean / Original" src={result?.original} filename="original.png" />}
            <ImagePanel label="Corrupted input" src={result?.input} filename="input.png" caption={result && describe(result.corruption)} />
            <ImagePanel label="Restored output" src={result?.output} filename={`restored_${mode}.png`} />
            {result?.original && <ImagePanel label="|Error| map" src={result?.error_map} filename="error_map.png" caption="mean absolute error vs clean" />}
          </div>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Inference time" value={result ? `${result.inference_ms.toFixed(1)} ms` : null} hint={result && `round-trip ${result.roundtrip_ms.toFixed(0)} ms`} accent="text-indigo-300" />
            <Stat label="PSNR (in → out)" value={m ? `${m.psnr_input ?? "∞"} → ${m.psnr_output ?? "∞"}` : null} hint="dB, vs clean" />
            <Stat label="SSIM (in → out)" value={m ? `${m.ssim_input} → ${m.ssim_output}` : null} hint="vs clean" />
            <Stat label="Corruption" value={result ? pretty(result.corruption.type?.split(" ")[0]) : null} hint={result?.corruption?.severity} />
          </div>

          {mode === "hard" && result && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Corruption classifier" subtitle="softmax probabilities p = C(x̃)">
                <Bars values={result.probabilities} highlight={result.predicted} />
              </Card>
              <Card title="Routing decision" subtitle="r = argmax p → one specialist (clean → identity bypass)">
                <div className="space-y-3 text-sm">
                  <Row k="Predicted corruption" v={<Badge>{pretty(result.predicted)}</Badge>} />
                  <Row k="Expert used" v={<Badge tone="violet">{result.expert}</Badge>} />
                  <Row
                    k="True corruption"
                    v={
                      result.true_corruption ? (
                        <span className={result.true_corruption === result.predicted ? "text-emerald-300" : "text-rose-300"}>
                          {pretty(result.true_corruption)} {result.true_corruption === result.predicted ? "✓ correct" : "✗ misrouted"}
                        </span>
                      ) : (
                        <span className="text-slate-500">unknown (uploaded)</span>
                      )
                    }
                  />
                  <Row k="Classifier time" v={`${result.timing_ms.classifier} ms`} />
                  <Row k="Expert time" v={`${result.timing_ms.expert} ms`} />
                </div>
              </Card>
            </div>
          )}

          {mode === "moe" && result && (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Routing weights" subtitle="w = softmax(G(x̃) / τ)">
                <Bars values={result.weights} highlight={result.dominant} />
              </Card>
              <Card title="Expert contribution" subtitle="x̂ = w₀x̃ + w₁A_salt + w₂A_blur + w₃A_occ">
                <StackedBar values={result.weights} />
                <div className="mt-5 space-y-3 text-sm">
                  <Row k="Strongest contributor" v={<Badge>{pretty(result.dominant)}</Badge>} />
                  <Row k="Ranking" v={<span className="text-xs text-slate-300">{result.ranking.map(pretty).join(" › ")}</span>} />
                  <Row
                    k="Routing entropy"
                    v={`${result.entropy} / 1.386 ${result.entropy < 0.4 ? "(sharp)" : result.entropy > 1.0 ? "(distributed)" : "(mixed)"}`}
                  />
                  <Row k="True corruption" v={result.true_corruption ? pretty(result.true_corruption) : "unknown (uploaded)"} />
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-800/70 pb-2 last:border-0">
      <span className="text-slate-400">{k}</span>
      <span className="text-right text-slate-200">{v}</span>
    </div>
  );
}

function Badge({ children, tone = "indigo" }) {
  const c = tone === "violet" ? "bg-violet-500/15 text-violet-200 border-violet-400/40" : "bg-indigo-500/15 text-indigo-200 border-indigo-400/40";
  return <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${c}`}>{children}</span>;
}
