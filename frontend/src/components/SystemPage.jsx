import { useEffect, useState } from "react";
import { getSystem } from "../api";
import { Card, ErrorBox, Stat } from "./ui";

export default function SystemPage() {
  const [sys, setSys] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    getSystem().then(setSys).catch((e) => setError(e.message));
  }, []);
  const verify = Object.fromEntries((sys?.onnx_verification || []).map((v) => [v.file, v]));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">System information</h1>
        <p className="mt-1 text-sm text-slate-400">Backend runtime, loaded ONNX models and PyTorch ↔ ONNX consistency checks.</p>
      </div>
      <ErrorBox error={error} />
      {sys && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="ONNX Runtime" value={sys.onnxruntime} />
            <Stat label="Python" value={sys.python} />
            <Stat label="CPU cores" value={sys.cpu_count} />
            <Stat label="Models loaded" value={`${Object.keys(sys.models).length} / ${Object.keys(sys.models).length + Object.keys(sys.missing).length}`} />
          </div>
          <Card title="ONNX models" subtitle="max |PyTorch − ONNX Runtime| measured at export time on real test images">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-[11px] uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">File</th>
                    <th className="py-2 pr-4">Size</th>
                    <th className="py-2 pr-4">Inputs</th>
                    <th className="py-2 pr-4">Outputs</th>
                    <th className="py-2 pr-4">Max abs diff</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {Object.entries(sys.models).map(([k, m]) => (
                    <tr key={k}>
                      <td className="py-2 pr-4 font-medium text-slate-200">{k}</td>
                      <td className="py-2 pr-4 text-slate-400">{m.file}</td>
                      <td className="py-2 pr-4 tabular-nums text-slate-400">{m.size_mb} MB</td>
                      <td className="py-2 pr-4 text-xs text-slate-400">{m.inputs.map((i) => `${i.name} ${JSON.stringify(i.shape)}`).join(", ")}</td>
                      <td className="py-2 pr-4 text-xs text-slate-400">{m.outputs.join(", ")}</td>
                      <td className="py-2 pr-4 tabular-nums">
                        {verify[m.file] ? (
                          <span className={verify[m.file].passed ? "text-emerald-300" : "text-rose-300"}>
                            {verify[m.file].max_abs_diff.toExponential(2)} {verify[m.file].passed ? "✓" : "✗"}
                          </span>
                        ) : (
                          <span className="text-slate-600">n/a</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {Object.entries(sys.missing).map(([k, e]) => (
                    <tr key={k}>
                      <td className="py-2 pr-4 font-medium text-rose-300">{k}</td>
                      <td colSpan={5} className="py-2 pr-4 text-xs text-rose-300/80">{e}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Corruption presets" subtitle="fixed test-time severities used by the workspaces">
            <pre className="overflow-x-auto text-xs text-slate-400">{JSON.stringify(sys.corruption_presets, null, 2)}</pre>
          </Card>
          <Card title="Host">
            <p className="text-sm text-slate-400">{sys.platform}</p>
          </Card>
        </>
      )}
    </div>
  );
}
