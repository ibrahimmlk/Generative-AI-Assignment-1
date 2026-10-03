// Shared building blocks, styled after the Google Stitch design (design/stitch/).

export const Icon = ({ name, className = "" }) => <span className={`material-symbols-outlined ${className}`}>{name}</span>;

export function Card({ title, subtitle, icon, children, className = "", right }) {
  return (
    <section className={`rounded-xl bg-surface-container-low/90 p-6 shadow-md backdrop-blur-xl ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div className="flex items-start gap-2">
            {icon && <Icon name={icon} className="mt-0.5 text-xl text-primary" />}
            <div>
              {title && <h2 className="font-display text-lg font-medium tracking-tight text-on-surface">{title}</h2>}
              {subtitle && <p className="mt-0.5 text-xs text-on-surface-variant">{subtitle}</p>}
            </div>
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ eyebrow, title, description, right }) {
  return (
    <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
      <div>
        <div className="mb-1 flex items-center gap-1.5 font-mono text-[13px] text-tertiary">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-tertiary" />
          <span>{eyebrow}</span>
        </div>
        <h1 className="font-display text-[32px] font-semibold leading-10 tracking-tight text-on-surface">{title}</h1>
        <p className="mt-0.5 max-w-2xl text-sm text-on-surface-variant">{description}</p>
      </div>
      {right}
    </div>
  );
}

export function download(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

const TAG_TONES = {
  neutral: "bg-surface-container-lowest text-outline",
  error: "bg-error-container/30 text-error",
  tertiary: "bg-tertiary-container/30 text-tertiary",
  primary: "bg-primary-container/20 text-primary-fixed",
};

export function ImagePanel({ label, src, filename, caption, tag, tagTone = "neutral", footer, highlight, placeholder = "Run the model to see the result" }) {
  return (
    <div className="group flex flex-col overflow-hidden rounded-xl bg-surface-container-low shadow-md">
      <div className="flex items-center justify-between bg-surface-container-high/80 px-4 py-2">
        <span className="font-display text-[15px] font-medium text-on-surface">{label}</span>
        {tag && <span className={`pill ${TAG_TONES[tagTone]}`}>{tag}</span>}
      </div>
      <div className="relative flex aspect-square items-center justify-center overflow-hidden bg-surface-container-lowest">
        {src ? (
          <img src={src} alt={label} className="pixelated h-full w-full object-contain transition-transform duration-500 group-hover:scale-105" />
        ) : (
          <div className="flex flex-col items-center gap-2 px-6 text-center text-xs text-outline">
            <Icon name="image" className="text-3xl" />
            {placeholder}
          </div>
        )}
        {src && caption && (
          <div className="absolute bottom-2 left-2 rounded bg-surface-container-lowest/80 px-1.5 py-0.5 font-mono text-[11px] text-on-surface backdrop-blur-md">
            {caption}
          </div>
        )}
      </div>
      <div className="flex items-center justify-between bg-surface-container px-4 py-2 font-mono text-[12px] text-on-surface-variant">
        <span className={highlight ? "text-tertiary" : ""}>{footer || "128 × 128 px"}</span>
        <button
          disabled={!src}
          onClick={() => download(src, filename || `${label.toLowerCase().replace(/\W+/g, "_")}.png`)}
          className={`rounded-lg p-1 transition disabled:opacity-30 ${
            highlight ? "bg-primary-container text-on-primary-container hover:opacity-90" : "text-on-surface hover:bg-surface-container-high"
          }`}
          title="Download PNG"
        >
          <Icon name="download" className="text-base" />
        </button>
      </div>
    </div>
  );
}

export function Stat({ label, value, unit, hint, icon, accent = "text-on-surface", iconColor = "text-primary" }) {
  return (
    <div className="flex flex-col justify-between rounded-xl bg-surface-container-low/80 p-4 shadow-sm backdrop-blur-md">
      <div className="flex items-center justify-between text-outline">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-wider">{label}</span>
        {icon && <Icon name={icon} className={`text-lg ${iconColor}`} />}
      </div>
      <div className={`my-2 font-mono text-[24px] font-semibold leading-8 tracking-tight tabular-nums ${accent}`}>
        {value ?? "—"} {value != null && unit && <span className="font-sans text-sm font-normal text-on-surface-variant">{unit}</span>}
      </div>
      {hint ? <span className="pill w-fit bg-surface-container-high normal-case tracking-normal text-tertiary">{hint}</span> : <span className="h-[18px]" />}
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
  blur: "bg-tertiary",
  occlusion: "bg-secondary",
};

export function Bars({ values, highlight }) {
  return (
    <div className="space-y-3">
      {Object.entries(values).map(([k, v]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between font-mono text-[12px]">
            <span className={k === highlight ? "font-semibold text-on-surface" : "text-on-surface-variant"}>
              {pretty(k)} {k === highlight && <Icon name="check_circle" className="ml-1 align-[-3px] text-sm text-tertiary" />}
            </span>
            <span className="tabular-nums text-on-surface">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-surface-container-highest">
            <div
              className={`h-full rounded-full transition-all duration-500 ${BRANCH_COLORS[k]} ${
                k === highlight ? "shadow-[0_0_8px_rgba(76,215,246,0.5)]" : "opacity-50"
              }`}
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
      <div className="flex h-4 overflow-hidden rounded-full bg-surface-container-highest">
        {Object.entries(values).map(([k, v]) => (
          <div key={k} className={`${BRANCH_COLORS[k]} h-full`} style={{ width: `${v * 100}%` }} title={`${pretty(k)} ${(v * 100).toFixed(1)}%`} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 font-mono text-[11px] text-on-surface-variant">
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
  return (
    <div className="flex items-start gap-2 rounded-xl bg-error-container/30 px-4 py-3 text-sm text-error">
      <Icon name="error" className="text-lg" />
      {error}
    </div>
  );
}

export function Spinner() {
  return <Icon name="progress_activity" className="animate-spin text-xl" />;
}
