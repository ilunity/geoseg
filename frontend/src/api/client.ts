// frontend/src/api/client.ts
// const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const BASE = import.meta.env.VITE_API_URL ?? ''

export interface HealthResponse {
  ok:      boolean
  loaded:  string[]
  failed:  Record<string, string>
  custom:  CustomModelMeta[]
}

export interface CustomModelClass {
  id:      number
  name:    string
  name_ru: string
  color:   [number, number, number]  // BGR
}

export interface CustomModelMeta {
  key:          string
  name:         string
  description:  string
  num_classes:  number
  model_type:   'binary' | 'multiclass'
  classes:      CustomModelClass[]
  encoder_name: string
  img_size:     number
  mean:         number[]
  std:          number[]
  created_at:   string
  file:         string
  loaded?:      boolean
  error?:       string | null
}

export interface InspectResult {
  num_classes:  number | null
  encoder_name: string
  img_size:     number
  mean:         number[]
  std:          number[]
}

export interface SegResult {
  terrain:        Record<string, unknown> | null
  coverage:       Record<string, number>
  models_used:    string[]
  custom_results: CustomSegResult[]
  log:            string[]
  ms:             number
  size:           [number, number]
  images: {
    mask:     string
    overlay:  string
    contours: string
  }
}

export interface CustomSegResult {
  key:        string
  name:       string
  coverage:   Record<string, number>
  classes:    CustomModelClass[]
  model_type: string
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  const r = await fetch(`${BASE}/api/health`)
  if (!r.ok) throw new Error('Backend unavailable')
  const data = await r.json()
  return {
    ok:     data.ok     ?? false,
    loaded: data.loaded ?? [],
    failed: data.failed ?? {},
    custom: data.custom ?? [],
  }
}

// ─── Segment ─────────────────────────────────────────────────────────────────

export async function segment(
  file:     File,
  mode:     string,
  selected: string[],
): Promise<SegResult> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('mode', mode)
  fd.append('selected_models', selected.join(','))
  const r = await fetch(`${BASE}/api/segment`, { method: 'POST', body: fd })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || `HTTP ${r.status}`)
  }
  const data = await r.json()
  // Защита: гарантируем наличие всех полей даже если бэкенд старый
  return {
    terrain:        data.terrain        ?? null,
    coverage:       data.coverage       ?? {},
    models_used:    data.models_used    ?? [],
    custom_results: data.custom_results ?? [],
    log:            data.log            ?? [],
    ms:             data.ms             ?? 0,
    size:           data.size           ?? [0, 0],
    images:         data.images         ?? { mask: '', overlay: '', contours: '' },
  }
}

// ─── Export ──────────────────────────────────────────────────────────────────

export async function exportData(fmt: 'geojson' | 'zip'): Promise<void> {
  const r = await fetch(`${BASE}/api/export/${fmt}`, { method: 'POST' })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || `HTTP ${r.status}`)
  }
  const blob = await r.blob()
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = fmt === 'geojson' ? 'segmentation.geojson' : 'geoseg_export.zip'
  a.click()
  URL.revokeObjectURL(url)
}

// ─── Custom models ────────────────────────────────────────────────────────────

export async function inspectModel(file: File): Promise<InspectResult> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch(`${BASE}/api/models/inspect`, { method: 'POST', body: fd })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || `HTTP ${r.status}`)
  }
  return r.json()
}

export interface UploadModelPayload {
  file:         File
  name:         string
  description:  string
  num_classes:  number
  model_type:   'binary' | 'multiclass'
  classes:      CustomModelClass[]
  encoder_name: string
  img_size:     number
  mean?:        number[]
  std?:         number[]
}

export async function uploadModel(
  payload: UploadModelPayload,
): Promise<{ ok: boolean; meta: CustomModelMeta; msg: string }> {
  const fd = new FormData()
  fd.append('file',         payload.file)
  fd.append('name',         payload.name)
  fd.append('description',  payload.description)
  fd.append('num_classes',  String(payload.num_classes))
  fd.append('model_type',   payload.model_type)
  fd.append('classes_json', JSON.stringify(payload.classes))
  fd.append('encoder_name', payload.encoder_name)
  fd.append('img_size',     String(payload.img_size))
  if (payload.mean) fd.append('mean_json', JSON.stringify(payload.mean))
  if (payload.std)  fd.append('std_json',  JSON.stringify(payload.std))

  const r = await fetch(`${BASE}/api/models/upload`, { method: 'POST', body: fd })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || `HTTP ${r.status}`)
  }
  return r.json()
}

export async function listCustomModels(): Promise<CustomModelMeta[]> {
  const r = await fetch(`${BASE}/api/models/custom`)
  if (!r.ok) throw new Error('Ошибка получения списка моделей')
  return r.json()
}

export async function deleteCustomModel(key: string): Promise<void> {
  const r = await fetch(`${BASE}/api/models/custom/${key}`, { method: 'DELETE' })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || `HTTP ${r.status}`)
  }
}