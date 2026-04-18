const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export type ApiHealth = {
  status: string
  service: string
  version: string
  modelReady: boolean
}

export type LocalSessionImage = {
  id: string
  name: string
  relativePath: string
  annotationCount: number
  annotationFormat?: string | null
}

export type LocalSessionResponse = {
  cancelled: boolean
  sessionId?: string
  sessionLabel?: string
  rootPath?: string
  images?: LocalSessionImage[]
}

export type LocalSessionJobStartResponse = {
  cancelled: boolean
  jobId?: string
}

export type LocalAnnotation = {
  id: string
  label: string
  difficult: boolean
  x: number
  y: number
  width: number
  height: number
}

export type LocalImageAnnotationsResponse = {
  format?: string | null
  count: number
  annotations: LocalAnnotation[]
}

export type LocalSessionJobResponse = {
  jobId: string
  status: 'running' | 'completed' | 'failed'
  phase: 'indexing' | 'loading' | 'completed' | 'failed'
  processedImages: number
  totalImages: number
  sessionRevision: number
  session?: LocalSessionResponse
  error?: string | null
}

export type RecentDatasetEntry = {
  path: string
  label: string
}

export type PersistedSessionStatePayload = {
  sourceKind: 'image' | 'dataset'
  sourcePath: string
  currentImageRelativePath?: string | null
}

export type AppStateResponse = {
  sidebarVisible: boolean
  recentDatasets: RecentDatasetEntry[]
  sessionState?: PersistedSessionStatePayload | null
  hotkeys?: Record<string, string[]> | null
}

export async function fetchApiHealth(signal?: AbortSignal) {
  return requestJson<ApiHealth>('/api/health', { signal })
}

export async function fetchPredefinedClasses(signal?: AbortSignal) {
  return requestJson<string[]>('/api/classes', { signal })
}

export async function fetchAppState(signal?: AbortSignal) {
  return requestJson<AppStateResponse>('/api/app-state', { signal })
}

export async function updateAppState(payload: Partial<AppStateResponse>) {
  return requestJson<AppStateResponse>('/api/app-state', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export async function openLocalImage() {
  return requestJson<LocalSessionResponse>('/api/local/sessions/open-image', {
    method: 'POST',
  })
}

export async function openLocalImagePath(path: string) {
  return requestJson<LocalSessionResponse>('/api/local/sessions/open-image-path', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path }),
  })
}

export async function openLocalDirectory() {
  return requestJson<LocalSessionJobStartResponse>('/api/local/sessions/open-directory', {
    method: 'POST',
  })
}

export async function openLocalDirectoryPath(path: string) {
  return requestJson<LocalSessionResponse>(
    '/api/local/sessions/open-directory-path',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path }),
    },
  )
}

export async function openLocalDirectoryPathJob(path: string) {
  return requestJson<LocalSessionJobStartResponse>(
    '/api/local/sessions/open-directory-path-job',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path }),
    },
  )
}

export async function fetchLocalSessionJob(jobId: string, afterRevision = 0) {
  return requestJson<LocalSessionJobResponse>(
    `/api/local/session-jobs/${jobId}?after_revision=${afterRevision}`,
  )
}

export async function fetchLocalAnnotations(sessionId: string, imageId: string) {
  return requestJson<LocalImageAnnotationsResponse>(
    `/api/local/sessions/${sessionId}/annotations/${imageId}`,
  )
}

export function buildLocalImageUrl(sessionId: string, imageId: string) {
  void sessionId
  return `${API_BASE_URL}/api/local/images/${imageId}`
}

async function requestJson<T>(path: string, init?: RequestInit) {
  const response = await fetch(`${API_BASE_URL}${path}`, init)

  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(detail || `API request failed: ${response.status}`)
  }

  return (await response.json()) as T
}

async function readErrorDetail(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string }
    return payload.detail ?? ''
  } catch {
    return ''
  }
}
