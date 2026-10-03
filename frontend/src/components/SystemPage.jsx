import { useEffect, useState } from "react";
import { getSystem } from "../api";
import { Card, ErrorBox, PageHeader, Stat } from "./ui";

export default function SystemPage() {
  const [sys, setSys] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    getSystem().then(setSys).catch((e) => setError(e.message));
  }, []);
  const verify = Object.fromEntries((sys?.onnx_verification || []).map((v) => [v.file, v]));

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="RUNTIME // FASTAPI + ONNX RUNTIME" title="System Information" description="Backend runtime, loaded ONNX models and the PyTorch ↔ ONNX Runtime consistency checks recorded at export time." />
      <ErrorBox error={error} />
      {sys && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="ONNX Runtime" icon="developer_board" value={sys.onnxruntime} />
            <Stat label="Python" icon="code" value={sys.python} />
            <Stat label="CPU cores" icon="memory" value={sys.cpu_count} />
            <Stat label="Models loaded" icon="database" value={`${Object.keys(sys.models).length} / ${Object.keys(sys.models).length + Object.keys(sys.missing).length}`} />
          </div>
          <Card icon="verified" title="ONNX models" subtitle="max |PyTorch − ONNX Runtime| measured at export time on real test images">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-[11px] uppercase tracking-wide text-outline">
                  <tr>
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">File</th>
                    <th className="py-2 pr-4">Size</th>
                    <th className="py-2 pr-4">Inputs</th>
                    <th className="py-2 pr-4">Outputs</th>
                    <th className="py-2 pr-4">Max abs diff</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/40">
                  {Object.entries(sys.models).map(([k, m]) => (
                    <tr key={k}>
                      <td className="py-2 pr-4 font-medium text-on-surface">{k}</td>
                      <td className="py-2 pr-4 text-on-surface-variant">{m.file}</td>
                      <td className="py-2 pr-4 tabular-nums text-on-surface-variant">{m.size_mb} MB</td>
                      <td className="py-2 pr-4 text-xs text-on-surface-variant">{m.inputs.map((i) => `${i.name} ${JSON.stringify(i.shape)}`).join(", ")}</td>
                      <td className="py-2 pr-4 text-xs text-on-surface-variant">{m.outputs.join(", ")}</td>
                      <td className="py-2 pr-4 tabular-nums">
                        {verify[m.file] ? (
                          <span className={verify[m.file].passed ? "text-tertiary" : "text-error"}>
                            {verify[m.file].max_abs_diff.toExponential(2)} {verify[m.file].passed ? "✓" : "✗"}
                          </span>
                        ) : (
                          <span className="text-outline">n/a</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {Object.entries(sys.missing).map(([k, e]) => (
                    <tr key={k}>
                      <td className="py-2 pr-4 font-medium text-error">{k}</td>
                      <td colSpan={5} className="py-2 pr-4 text-xs text-error/80">{e}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card icon="tune" title="Corruption presets" subtitle="fixed test-time severities used by the workspaces">
            <pre className="overflow-x-auto text-xs text-on-surface-variant">{JSON.stringify(sys.corruption_presets, null, 2)}</pre>
          </Card>
          <Card icon="dns" title="Host">
            <p className="text-sm text-on-surface-variant">{sys.platform}</p>
          </Card>
        </>
      )}
    </div>
  );
}
