const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail || `API request failed: ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

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

export type LocalSessionClassInfo = {
  label: string
  sourceClassIndex?: number | null
  hasUnknownClass?: boolean
  annotationCount: number
  imageCount: number
}

export type LocalSessionAnnotationFormatInfo = {
  format: string
  imageCount: number
}

export type LocalSessionInfoResponse = {
  sessionId: string
  sessionLabel: string
  rootPath: string
  imageCount: number
  annotatedImageCount: number
  emptyImageCount: number
  annotationCount: number
  difficultAnnotationCount: number
  unknownClassCount: number
  annotationFormats: LocalSessionAnnotationFormatInfo[]
  classes: LocalSessionClassInfo[]
}

export type LocalSessionNotification = {
  sequence: number
  type: string
  tone: 'info' | 'success' | 'error'
  message: string
  imageId?: string | null
  relativePath?: string | null
  quarantinePath?: string | null
}

export type LocalSessionNotificationsResponse = {
  sequence: number
  notifications: LocalSessionNotification[]
  session?: LocalSessionResponse | null
}

export type LocalSessionJobStartResponse = {
  cancelled: boolean
  jobId?: string
}

export type LocalSessionImageUploadResponse = {
  image: LocalSessionImage
  session: LocalSessionResponse
}

export type LocalSessionImagesUploadResponse = {
  images: LocalSessionImage[]
  session: LocalSessionResponse
  skippedExistingFilenames?: string[]
}

export type LocalSessionImageDeleteResponse = {
  cancelled: boolean
  sessionId?: string
  sessionLabel?: string
  rootPath?: string
  deletedImageIds: string[]
  deletedImages?: LocalSessionImage[]
  remainingImageCount: number
}

export type LocalAnnotation = {
  id: string
  label: string
  sourceClassIndex?: number | null
  hasUnknownClass?: boolean
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
  removedDuplicateCount?: number
}

export type SaveLocalAnnotationsResponse = {
  format?: string | null
  count: number
  annotations?: LocalAnnotation[] | null
  removedDuplicateCount?: number
  savedAt?: string | null
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

export type LocalDuplicateSearchJobStartResponse = {
  jobId: string
}

export type LocalDuplicateSearchMatch = {
  id: string
  leftImageId: string
  leftName: string
  leftRelativePath: string
  rightImageId: string
  rightName: string
  rightRelativePath: string
  similarityPercent: number
  relativeTransform: string
}

export type LocalDuplicateSearchJobResponse = {
  jobId: string
  status: 'running' | 'completed' | 'failed'
  phase: 'hashing' | 'comparing' | 'completed' | 'failed'
  processedImages: number
  totalImages: number
  processedPairs: number
  totalPairs: number
  minimumSimilarityPercent: number
  revision: number
  matchCount: number
  matches?: LocalDuplicateSearchMatch[] | null
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

export type SamPromptPairPayload = {
  prompt?: string | null
  label?: string | null
}

export type SamSettingsPayload = {
  entries?: SamPromptPairPayload[] | null
  scoreThreshold?: string | null
  maxResults?: string | null
}

export type AppStateResponse = {
  sidebarVisible: boolean
  recentDatasets: RecentDatasetEntry[]
  sessionState?: PersistedSessionStatePayload | null
  hotkeys?: Record<string, string[]> | null
  projectClassesByRootPath?: Record<string, string[]> | null
  samSettings?: SamSettingsPayload | null
}

export type CacheDbTableCounts = {
  appSettings: number
  serviceSecrets: number
  datasetManifests: number
  datasetImages: number
  annotationCache: number
}

export type CacheDbSummary = {
  dbPath: string
  exists: boolean
  dbBytes: number
  walBytes: number
  shmBytes: number
  totalBytes: number
  tableCounts: CacheDbTableCounts
}

export type CacheDbAction =
  | 'clear-session-history'
  | 'clear-dataset-cache'
  | 'clear-annotation-cache'
  | 'reset-project-cache-db'
  | 'compact-database'

export type PluginModelState = {
  filename: string
  provider: string
  repoId?: string | null
  requiresAuth: boolean
  expectedSizeBytes?: number | null
  accessUrl?: string | null
  docsUrl?: string | null
  isInstalled: boolean
  installedBytes?: number | null
  installedAt?: string | null
  path?: string | null
}

export type PluginDownloadState = {
  status: 'idle' | 'running' | 'completed' | 'failed'
  downloadedBytes: number
  totalBytes?: number | null
  error?: string | null
  startedAt?: string | null
  finishedAt?: string | null
}

export type PluginRuntimeInstallProfile = 'auto' | 'cuda' | 'cpu'

export type PluginRuntimeInstallState = {
  status: 'idle' | 'running' | 'completed' | 'failed'
  requestedProfile?: PluginRuntimeInstallProfile | null
  resolvedProfile?: Exclude<PluginRuntimeInstallProfile, 'auto'> | null
  step?: string | null
  stepStartedAt?: string | null
  log?: string | null
  message?: string | null
  error?: string | null
  startedAt?: string | null
  finishedAt?: string | null
}

export type PluginRuntimeState = {
  status: 'ready' | 'missing-model' | 'missing-runtime' | 'error'
  device?: string | null
  message?: string | null
  supportsAutoAnnotate: boolean
  install: PluginRuntimeInstallState
}

export type PluginInfo = {
  id: string
  name: string
  version: string
  summary: string
  description: string
  capabilities: string[]
  integrationTarget: string
  download: PluginDownloadState
  runtime: PluginRuntimeState
  model: PluginModelState
}

type PluginRuntimeResponse = Omit<PluginRuntimeState, 'install'> & {
  install?: Partial<PluginRuntimeInstallState> | null
}

type PluginInfoResponse = Omit<PluginInfo, 'download' | 'runtime'> & {
  download?: Partial<PluginDownloadState> | null
  runtime?: Partial<PluginRuntimeResponse> | null
}

export type PluginAutoAnnotateRegion = {
  x: number
  y: number
  width: number
  height: number
}

export type PluginAutoAnnotateRequest = {
  sessionId?: string | null
  imageId: string
  prompt: string
  label?: string | null
  mode?: 'full-image' | 'selected-box'
  region?: PluginAutoAnnotateRegion | null
  scoreThreshold?: number
  maxResults?: number
}

export type PluginAutoAnnotateResult = {
  pluginId: string
  mode: 'full-image' | 'selected-box'
  prompt: string
  label: string
  annotationCount: number
  annotations: Array<
    PluginAutoAnnotateRegion & {
      label: string
      score: number
    }
  >
}

export type HuggingFaceAuthUser = {
  username?: string | null
  fullName?: string | null
  email?: string | null
  avatarUrl?: string | null
  organizations?: string[] | null
}

export type HuggingFaceAuthStatus = {
  provider: string
  isConfigured: boolean
  clientId?: string | null
  callbackUrl?: string | null
  scopes: string[]
  isAuthenticated: boolean
  isExpired: boolean
  hasUsableAccessToken: boolean
  authSource?: 'oauth' | null
  user?: HuggingFaceAuthUser | null
  expiresAt?: string | null
  savedAt?: string | null
}

export type HuggingFaceAuthStartResponse = {
  authorizationUrl: string
  redirectUri: string
  scopes: string[]
  clientId: string
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

export async function fetchCacheDbSummary(signal?: AbortSignal) {
  return requestJson<CacheDbSummary>('/api/cache-db', { signal })
}

export async function runCacheDbAction(action: CacheDbAction) {
  return requestJson<CacheDbSummary>('/api/cache-db/actions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ action }),
  })
}

export async function fetchPlugins(signal?: AbortSignal) {
  const plugins = await requestJson<PluginInfoResponse[]>('/api/plugins', {
    signal,
  })
  return plugins.map(normalizePluginInfo)
}

export async function downloadPluginModel(pluginId: string) {
  const plugin = await requestJson<PluginInfoResponse>(
    `/api/plugins/${pluginId}/download-model`,
    {
      method: 'POST',
    },
  )
  return normalizePluginInfo(plugin)
}

export async function installPluginRuntime(
  pluginId: string,
  profile: PluginRuntimeInstallProfile,
) {
  const plugin = await requestJson<PluginInfoResponse>(
    `/api/plugins/${pluginId}/install-runtime`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ profile }),
    },
  )
  return normalizePluginInfo(plugin)
}

function normalizePluginInfo(plugin: PluginInfoResponse): PluginInfo {
  return {
    ...plugin,
    download: {
      status: normalizePluginDownloadStatus(plugin.download?.status),
      downloadedBytes: plugin.download?.downloadedBytes ?? 0,
      totalBytes:
        plugin.download?.totalBytes ?? plugin.model.expectedSizeBytes ?? null,
      error: plugin.download?.error ?? null,
      startedAt: plugin.download?.startedAt ?? null,
      finishedAt: plugin.download?.finishedAt ?? null,
    },
    runtime: {
      status: normalizePluginRuntimeStatus(plugin.runtime?.status),
      device: plugin.runtime?.device ?? null,
      message: plugin.runtime?.message ?? null,
      supportsAutoAnnotate: Boolean(plugin.runtime?.supportsAutoAnnotate),
      install: {
        status: normalizePluginRuntimeInstallStatus(plugin.runtime?.install?.status),
        requestedProfile: normalizePluginRuntimeInstallProfile(
          plugin.runtime?.install?.requestedProfile,
        ),
        resolvedProfile: normalizePluginRuntimeResolvedProfile(
          plugin.runtime?.install?.resolvedProfile,
        ),
        step: plugin.runtime?.install?.step ?? null,
        stepStartedAt: plugin.runtime?.install?.stepStartedAt ?? null,
        log: plugin.runtime?.install?.log ?? null,
        message: plugin.runtime?.install?.message ?? null,
        error: plugin.runtime?.install?.error ?? null,
        startedAt: plugin.runtime?.install?.startedAt ?? null,
        finishedAt: plugin.runtime?.install?.finishedAt ?? null,
      },
    },
  }
}

function normalizePluginDownloadStatus(
  status: PluginDownloadState['status'] | undefined,
): PluginDownloadState['status'] {
  if (
    status === 'idle' ||
    status === 'running' ||
    status === 'completed' ||
    status === 'failed'
  ) {
    return status
  }
  return 'idle'
}

function normalizePluginRuntimeStatus(
  status: PluginRuntimeState['status'] | undefined,
): PluginRuntimeState['status'] {
  if (
    status === 'ready' ||
    status === 'missing-model' ||
    status === 'missing-runtime' ||
    status === 'error'
  ) {
    return status
  }
  return 'missing-runtime'
}

function normalizePluginRuntimeInstallStatus(
  status: PluginRuntimeInstallState['status'] | undefined,
): PluginRuntimeInstallState['status'] {
  if (
    status === 'idle' ||
    status === 'running' ||
    status === 'completed' ||
    status === 'failed'
  ) {
    return status
  }
  return 'idle'
}

function normalizePluginRuntimeInstallProfile(
  profile: PluginRuntimeInstallProfile | null | undefined,
): PluginRuntimeInstallProfile | null {
  if (profile === 'auto' || profile === 'cuda' || profile === 'cpu') {
    return profile
  }
  return null
}

function normalizePluginRuntimeResolvedProfile(
  profile: PluginRuntimeInstallState['resolvedProfile'] | null | undefined,
): PluginRuntimeInstallState['resolvedProfile'] {
  if (profile === 'cuda' || profile === 'cpu') {
    return profile
  }
  return null
}

export async function fetchHuggingFaceAuthStatus(signal?: AbortSignal) {
  return requestJson<HuggingFaceAuthStatus>('/api/hf-auth/status', { signal })
}

export async function startHuggingFaceAuth() {
  return requestJson<HuggingFaceAuthStartResponse>('/api/hf-auth/start', {
    method: 'POST',
  })
}

export async function logoutHuggingFaceAuth() {
  return requestJson<HuggingFaceAuthStatus>('/api/hf-auth/logout', {
    method: 'POST',
  })
}

export async function updateHuggingFaceAuthConfig(clientId: string | null) {
  return requestJson<HuggingFaceAuthStatus>('/api/hf-auth/config', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ clientId }),
  })
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

export async function uploadLocalSessionImage(sessionId: string, file: File) {
  const params = new URLSearchParams({
    filename: file.name,
  })

  return requestJson<LocalSessionImageUploadResponse>(
    `/api/local/sessions/${sessionId}/images/upload?${params.toString()}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
      },
      body: file,
    },
  )
}

export async function uploadLocalSessionImages(sessionId: string, files: File[]) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file, file.name)
  }

  return requestJson<LocalSessionImagesUploadResponse>(
    `/api/local/sessions/${sessionId}/images/bulk-upload`,
    {
      method: 'POST',
      body: formData,
    },
  )
}

export async function startLocalDuplicateSearch(
  sessionId: string,
  minimumSimilarityPercent: number,
) {
  return requestJson<LocalDuplicateSearchJobStartResponse>(
    `/api/local/sessions/${sessionId}/duplicate-search`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        minimumSimilarityPercent,
      }),
    },
  )
}

export async function fetchLocalDuplicateSearchJob(
  jobId: string,
  afterRevision = 0,
) {
  return requestJson<LocalDuplicateSearchJobResponse>(
    `/api/local/duplicate-search-jobs/${jobId}?after_revision=${afterRevision}`,
  )
}

export async function fetchLocalAnnotations(sessionId: string, imageId: string) {
  return requestJson<LocalImageAnnotationsResponse>(
    `/api/local/sessions/${sessionId}/annotations/${imageId}`,
  )
}

export async function fetchLocalSessionInfo(sessionId: string) {
  return requestJson<LocalSessionInfoResponse>(
    `/api/local/sessions/${sessionId}/info`,
  )
}

export async function fetchLocalSessionNotifications(
  sessionId: string,
  afterSequence = 0,
  signal?: AbortSignal,
) {
  return requestJson<LocalSessionNotificationsResponse>(
    `/api/local/sessions/${sessionId}/notifications?after_sequence=${afterSequence}`,
    { signal },
  )
}

export async function saveLocalAnnotations(
  sessionId: string,
  imageId: string,
  annotations: LocalAnnotation[],
  projectClasses: string[] = [],
) {
  return requestJson<SaveLocalAnnotationsResponse>(
    `/api/local/sessions/${sessionId}/annotations/${imageId}`,
    {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        annotations,
        projectClasses,
      }),
    },
  )
}

export async function deleteLocalSessionImage(
  sessionId: string,
  imageId: string,
) {
  return requestJson<LocalSessionImageDeleteResponse>(
    `/api/local/sessions/${sessionId}/images/${imageId}?response_mode=summary`,
    {
      method: 'DELETE',
    },
  )
}

export async function deleteLocalSessionImages(
  sessionId: string,
  imageIds: string[],
) {
  return requestJson<LocalSessionImageDeleteResponse>(
    `/api/local/sessions/${sessionId}/images/bulk-delete?response_mode=summary`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        imageIds,
      }),
    },
  )
}

export async function runPluginAutoAnnotate(
  pluginId: string,
  payload: PluginAutoAnnotateRequest,
) {
  return requestJson<PluginAutoAnnotateResult>(
    `/api/plugins/${pluginId}/auto-annotate`,
    {
      method: 'POST',
      timeoutMs: 90000,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    },
  )
}

export function buildLocalImageUrl(sessionId: string, imageId: string) {
  void sessionId
  return `${API_BASE_URL}/api/local/images/${imageId}`
}

type RequestJsonOptions = RequestInit & {
  timeoutMs?: number
}

async function requestJson<T>(path: string, init?: RequestJsonOptions) {
  const { timeoutMs, signal, ...requestInit } = init ?? {}
  const controller =
    timeoutMs !== undefined || signal !== undefined ? new AbortController() : null
  let didTimeout = false
  let timeoutId: number | null = null
  let abortListener: (() => void) | null = null

  if (controller && signal) {
    abortListener = () => controller.abort()
    if (signal.aborted) {
      controller.abort()
    } else {
      signal.addEventListener('abort', abortListener, { once: true })
    }
  }

  if (controller && timeoutMs && timeoutMs > 0) {
    timeoutId = window.setTimeout(() => {
      didTimeout = true
      controller.abort()
    }, timeoutMs)
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...requestInit,
      signal: controller?.signal ?? signal,
    })
  } catch (error) {
    if (didTimeout && timeoutMs) {
      throw new Error(
        `Request timed out after ${Math.max(1, Math.round(timeoutMs / 1000))}s`,
      )
    }
    throw error
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId)
    }
    if (signal && abortListener) {
      signal.removeEventListener('abort', abortListener)
    }
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new ApiError(response.status, detail)
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
