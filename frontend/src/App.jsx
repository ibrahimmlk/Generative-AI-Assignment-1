import { useEffect, useState } from "react";
import { getHealth } from "./api";
import RestorationWorkspace from "./components/RestorationWorkspace";
import SketchWorkspace from "./components/SketchWorkspace";
import SystemPage from "./components/SystemPage";

const PAGES = [
  {
    id: "universal",
    nav: "Universal Restoration",
    icon: "◎",
    tag: "Task 1",
    render: () => (
      <RestorationWorkspace
        key="universal"
        mode="universal"
        title="Universal Restoration"
        description="One convolutional denoising autoencoder with a compressed latent bottleneck restores clean, salt-and-pepper, blurred and occluded images without being told the corruption type."
      />
    ),
  },
  {
    id: "hard",
    nav: "Hard-Routed Restoration",
    icon: "⑂",
    tag: "Task 2",
    render: () => (
      <RestorationWorkspace
        key="hard"
        mode="hard"
        title="Hard-Routed Restoration"
        description="A CNN classifier predicts the corruption and routes the image to exactly one specialist autoencoder. Images predicted clean bypass all experts."
      />
    ),
  },
  {
    id: "moe",
    nav: "Soft Mixture-of-Experts",
    icon: "≋",
    tag: "Task 3",
    render: () => (
      <RestorationWorkspace
        key="moe"
        mode="moe"
        title="Soft Mixture-of-Experts Restoration"
        description="A gating network assigns a continuous weight to the identity branch and all three experts; the output is their weighted sum. Gate and experts were fine-tuned jointly."
      />
    ),
  },
  { id: "sketch", nav: "Face-to-Sketch Generator", icon: "✎", tag: "Task 4", render: () => <SketchWorkspace /> },
  { id: "system", nav: "System", icon: "⚙", tag: "Info", render: () => <SystemPage /> },
];

export default function App() {
  const [page, setPage] = useState(() => window.location.hash.slice(1) || "universal");
  const [health, setHealth] = useState(null);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    const load = () => getHealth().then(setHealth).catch(() => setHealth(null));
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

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top_left,rgba(99,102,241,0.12),transparent_50%)]">
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-slate-800 bg-slate-950/95 backdrop-blur transition-transform lg:translate-x-0 ${
          menu ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 px-6 py-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 font-bold text-white">G</div>
          <div>
            <div className="text-sm font-semibold text-white">GenAI Restoration Lab</div>
            <div className="text-[11px] text-slate-500">Generative AI · Assignment 1</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 px-3">
          {PAGES.map((p) => (
            <button
              key={p.id}
              onClick={() => go(p.id)}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${
                current.id === p.id ? "bg-indigo-500/15 text-indigo-100" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${current.id === p.id ? "bg-indigo-500 text-white" : "bg-slate-800"}`}>
                {p.icon}
              </span>
              <span className="flex-1">{p.nav}</span>
              <span className="text-[10px] uppercase tracking-wide text-slate-600">{p.tag}</span>
            </button>
          ))}
        </nav>
        <div className="m-3 rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-xs">
          <div className="flex items-center gap-2 font-medium text-slate-200">
            <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400" : health ? "bg-amber-400" : "bg-rose-500"}`} />
            Backend {ok ? "healthy" : health ? "degraded" : "unreachable"}
          </div>
          {health && (
            <div className="mt-2 space-y-1 text-slate-500">
              <div>ONNX Runtime {health.onnxruntime}</div>
              <div>
                {health.models_loaded.length} model(s) loaded
                {Object.keys(health.models_missing).length > 0 && `, ${Object.keys(health.models_missing).length} missing`}
              </div>
              <div>uptime {Math.round(health.uptime_s)} s</div>
            </div>
          )}
        </div>
      </aside>

      <div className="lg:pl-72">
        <div className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur lg:hidden">
          <button onClick={() => setMenu(!menu)} className="rounded-lg border border-slate-700 px-2.5 py-1 text-slate-300">
            ☰
          </button>
          <span className="text-sm font-medium text-slate-200">{current.nav}</span>
        </div>
        <main className="mx-auto max-w-7xl px-4 py-8 sm:px-8">{current.render()}</main>
      </div>
      {menu && <div className="fixed inset-0 z-20 bg-black/50 lg:hidden" onClick={() => setMenu(false)} />}
    </div>
  );
}
