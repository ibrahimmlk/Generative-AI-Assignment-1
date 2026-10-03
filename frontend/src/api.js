// Thin client for the FastAPI backend (served under /api by nginx or the Vite proxy).
async function request(path, options = {}) {
  const res = await fetch(`/api${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export const getHealth = () => request("/health");
export const getSystem = () => request("/system");
export const getSamples = () => request("/samples");
export const sampleUrl = (path) => `/api/sample-files/${path}`;

function form(fields) {
  const fd = new FormData();
  Object.entries(fields).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") fd.append(k, v);
  });
  return fd;
}

export const restore = (mode, fields) => request(`/restore/${mode}`, { method: "POST", body: form(fields) });
export const sketch = (fields) => request("/sketch", { method: "POST", body: form(fields) });
