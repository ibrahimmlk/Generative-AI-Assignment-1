// Small shared building blocks: cards, image panels, stat tiles, bar charts.

export function Card({ title, subtitle, children, className = "", right }) {
  return (
    <section className={`rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg shadow-black/20 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function download(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

export function ImagePanel({ label, src, filename, caption, placeholder = "Run the model to see the result" }) {
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</span>
        {src && (
          <button
            onClick={() => download(src, filename || `${label.toLowerCase().replace(/\W+/g, "_")}.png`)}
            className="rounded-md px-2 py-0.5 text-xs text-indigo-300 hover:bg-indigo-500/10"
            title="Download PNG"
          >
            ↓ PNG
          </button>
        )}
      </div>
      <div className="flex aspect-square items-center justify-center bg-[repeating-conic-gradient(#0f172a_0%_25%,#111827_0%_50%)] bg-[length:16px_16px]">
        {src ? (
          <img src={src} alt={label} className="pixelated h-full w-full object-contain" />
        ) : (
          <span className="px-6 text-center text-xs text-slate-600">{placeholder}</span>
        )}
      </div>
      {caption && <div className="border-t border-slate-800 px-3 py-1.5 text-[11px] text-slate-400">{caption}</div>}
    </div>
  );
}

export function Stat({ label, value, hint, accent }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold tabular-nums ${accent || "text-slate-100"}`}>{value ?? "—"}</div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

const PRETTY = {
  clean: "Clean",
  identity: "Identity (clean)",
  salt_pepper: "Salt & pepper",
  blur: "Gaussian blur",
  occlusion: "Occlusion",
};
export const pretty = (k) => PRETTY[k] || k;

export const BRANCH_COLORS = {
  clean: "bg-emerald-400",
  identity: "bg-emerald-400",
  salt_pepper: "bg-amber-400",
  blur: "bg-sky-400",
  occlusion: "bg-fuchsia-400",
};

export function Bars({ values, highlight }) {
  return (
    <div className="space-y-3">
      {Object.entries(values).map(([k, v]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between text-xs">
            <span className={k === highlight ? "font-semibold text-slate-100" : "text-slate-400"}>
              {pretty(k)} {k === highlight && <span className="ml-1 text-indigo-300">●</span>}
            </span>
            <span className="tabular-nums text-slate-300">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full rounded-full transition-all duration-500 ${BRANCH_COLORS[k]} ${k === highlight ? "" : "opacity-60"}`}
              style={{ width: `${Math.max(v * 100, 0.5)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function StackedBar({ values }) {
  return (
    <div>
      <div className="flex h-4 overflow-hidden rounded-full bg-slate-800">
        {Object.entries(values).map(([k, v]) => (
          <div key={k} className={`${BRANCH_COLORS[k]} h-full`} style={{ width: `${v * 100}%` }} title={`${pretty(k)} ${(v * 100).toFixed(1)}%`} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
        {Object.keys(values).map((k) => (
          <span key={k} className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${BRANCH_COLORS[k]}`} /> {pretty(k)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>;
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />;
}
