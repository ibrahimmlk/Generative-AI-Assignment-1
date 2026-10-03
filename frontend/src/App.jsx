import { useEffect, useState } from "react";
import { getHealth } from "./api";
import RestorationWorkspace from "./components/RestorationWorkspace";
import SketchWorkspace from "./components/SketchWorkspace";
import SystemPage from "./components/SystemPage";
import { Icon } from "./components/ui";

const PAGES = [
  {
    id: "universal",
    nav: "Universal Restoration",
    icon: "auto_awesome",
    crumb: "Task 1 · Denoising AE",
    render: () => (
      <RestorationWorkspace
        key="universal"
        mode="universal"
        eyebrow="TASK 1 // UNIVERSAL DENOISING AUTOENCODER"
        title="Universal Restoration"
        model="universal_ae.onnx"
        runLabel="Run Universal Restoration"
        description="One convolutional encoder–decoder with a compressed latent bottleneck restores clean, salt-and-pepper, blurred and occluded images without being told which corruption was applied."
      />
    ),
  },
  {
    id: "hard",
    nav: "Hard-Routed Restoration",
    icon: "alt_route",
    crumb: "Task 2 · Classifier + Specialists",
    render: () => (
      <RestorationWorkspace
        key="hard"
        mode="hard"
        eyebrow="TASK 2 // CLASSIFIER → SPECIALIST ROUTER"
        title="Hard-Routed Restoration"
        model="classifier + 3 specialists"
        runLabel="Classify & Route"
        description="A CNN classifier predicts the corruption and sends the image to exactly one specialist autoencoder. Images predicted clean bypass every expert."
      />
    ),
  },
  {
    id: "moe",
    nav: "Soft Mixture-of-Experts",
    icon: "memory",
    crumb: "Task 3 · Soft MoE",
    render: () => (
      <RestorationWorkspace
        key="moe"
        mode="moe"
        eyebrow="TASK 3 // JOINTLY TRAINED SOFT MIXTURE-OF-EXPERTS"
        title="Soft Mixture-of-Experts Restoration"
        model="soft_moe.onnx"
        runLabel="Run Soft MoE"
        description="A gating network assigns a continuous weight to the identity branch and to all three experts; the output is their weighted sum. Gate and experts were fine-tuned jointly."
      />
    ),
  },
  { id: "sketch", nav: "Face-to-Sketch Generator", icon: "draw", crumb: "Task 4 · Conditional GAN", render: () => <SketchWorkspace /> },
  { id: "system", nav: "System Information", icon: "monitor_heart", crumb: "Runtime", render: () => <SystemPage /> },
];

export default function App() {
  const [page, setPage] = useState(() => window.location.hash.slice(1) || "universal");
  const [health, setHealth] = useState(null);
  const [ping, setPing] = useState(null);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    const load = () => {
      const t0 = performance.now();
      getHealth()
        .then((h) => {
          setHealth(h);
          setPing(Math.round(performance.now() - t0));
        })
        .catch(() => setHealth(null));
    };
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);

  function go(id) {
    setPage(id);
    window.location.hash = id;
    setMenu(false);
  }

  const current = PAGES.find((p) => p.id === page) || PAGES[0];
  const ok = health?.status === "ok";
  const nLoaded = health?.models_loaded.length ?? 0;
  const nTotal = nLoaded + Object.keys(health?.models_missing || {}).length;

  return (
    <div className="min-h-screen bg-surface">
      <aside
        className={`fixed left-0 top-0 z-50 flex h-full w-72 flex-col justify-between bg-surface-container-low/95 shadow-[0_1px_8px_rgba(0,0,0,0.04)] backdrop-blur-2xl transition-transform lg:translate-x-0 ${
          menu ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex flex-col">
          <div className="flex items-center gap-4 p-6">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary-container to-secondary-container text-white">
              <Icon name="auto_fix_high" className="text-xl" />
            </div>
            <div className="flex min-w-0 flex-col">
              <span className="truncate font-display text-lg font-medium tracking-tight text-on-surface">GenAI Restoration Lab</span>
              <span className="truncate text-xs text-on-surface-variant">Generative AI · Assignment 1</span>
            </div>
          </div>
          <div className="px-6 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-outline">Neural Architectures</div>
          <nav className="mt-1 flex flex-col gap-1 px-4">
            {PAGES.map((p) => (
              <button
                key={p.id}
                onClick={() => go(p.id)}
                className={`flex items-center gap-4 rounded-lg px-4 py-2 text-left transition-all ${
                  current.id === p.id
                    ? "bg-primary-container font-display font-semibold text-on-primary-container shadow-[0_0_16px_-2px_rgba(99,102,241,0.2)]"
                    : "text-sm text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
                }`}
              >
                <Icon name={p.icon} className="text-xl" />
                <span className="truncate">{p.nav}</span>
              </button>
            ))}
          </nav>
        </div>
        <div className="flex flex-col gap-2 bg-surface-container-lowest/70 p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                {ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-tertiary opacity-75" />}
                <span className={`relative inline-flex h-2 w-2 rounded-full ${ok ? "bg-tertiary" : health ? "bg-amber-400" : "bg-error"}`} />
              </span>
              <span className="font-mono text-[13px] text-on-surface">{ok ? "System Online" : health ? "Degraded" : "Backend Offline"}</span>
            </div>
            <span className="font-mono text-[13px] text-outline">{ping != null ? `${ping}ms` : "—"}</span>
          </div>
          <div className="flex items-center justify-between gap-1">
            <span className="pill bg-surface-container-highest text-tertiary">ONNX RT {health?.onnxruntime || "?"} (CPU)</span>
            <span className="font-mono text-[11px] font-medium text-on-surface-variant">
              Models: {nLoaded}/{nTotal || 7}
            </span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-surface-container-highest">
            <div className="h-full rounded-full bg-tertiary shadow-[0_0_8px_rgba(76,215,246,0.5)]" style={{ width: `${nTotal ? (nLoaded / nTotal) * 100 : 0}%` }} />
          </div>
        </div>
      </aside>

      <div className="lg:pl-72">
        <header className="sticky top-0 z-40 bg-surface/85 shadow-[0_1px_8px_rgba(0,0,0,0.04)] backdrop-blur-xl">
          <div className="flex h-16 items-center justify-between px-4 sm:px-6">
            <div className="flex items-center gap-2 font-mono text-[13px]">
              <button onClick={() => setMenu(!menu)} className="mr-2 rounded-lg p-1 text-on-surface-variant hover:bg-surface-container-high lg:hidden">
                <Icon name="menu" />
              </button>
              <span className="hidden text-on-surface-variant sm:inline">Lab Core</span>
              <span className="hidden text-outline sm:inline">/</span>
              <span className="font-medium tracking-wide text-primary">{current.crumb}</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden items-center gap-2 rounded-full bg-surface-container-low px-4 py-1 xl:flex">
                <Icon name="developer_board" className="text-sm text-tertiary" />
                <span className="font-mono text-[13px] text-on-surface-variant">ONNX Runtime · CPUExecutionProvider · FP32</span>
              </div>
              <div className="hidden items-center gap-2 rounded-full bg-surface-container-low px-4 py-1 md:flex">
                <Icon name="database" className="text-sm text-primary" />
                <span className="font-mono text-[13px] text-on-surface-variant">
                  uptime <span className="font-semibold text-on-surface">{health ? Math.round(health.uptime_s) : "—"}</span> s
                </span>
              </div>
              <a
                href="/api/docs"
                onClick={(e) => {
                  e.preventDefault();
                  window.open(`${window.location.protocol}//${window.location.hostname}:8000/docs`, "_blank");
                }}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
              >
                <Icon name="menu_book" className="text-base" />
                <span className="hidden sm:inline">API Docs</span>
              </a>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6">{current.render()}</main>
      </div>
      {menu && <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setMenu(false)} />}
    </div>
  );
}
