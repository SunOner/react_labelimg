import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from 'react'
import './App.css'
import { AnnotationViewport } from './components/AnnotationViewport'
import { VirtualFileList } from './components/VirtualFileList'
import { AppButton } from './components/ui/AppButton'
import { ConfirmDialog } from './components/ui/ConfirmDialog'
import { MenuItemButton } from './components/ui/MenuItemButton'
import {
  buildLocalImageUrl,
  deleteLocalSessionImage,
  downloadPluginModel,
  fetchAppState,
  fetchApiHealth,
  fetchHuggingFaceAuthStatus,
  fetchLocalAnnotations,
  fetchLocalSessionJob,
  fetchPlugins,
  fetchPredefinedClasses,
  installPluginRuntime,
  logoutHuggingFaceAuth,
  openLocalDirectory,
  openLocalImage,
  openLocalImagePath,
  openLocalDirectoryPathJob,
  runPluginAutoAnnotate,
  saveLocalAnnotations,
  startHuggingFaceAuth,
  updateHuggingFaceAuthConfig,
  updateAppState,
  type HuggingFaceAuthStatus,
  type LocalSessionJobResponse,
  type LocalSessionResponse,
  type PluginAutoAnnotateResult,
  type PluginDownloadState,
  type PluginInfo,
  type PluginRuntimeInstallState,
  type PluginRuntimeInstallProfile,
} from './lib/api'
import {
  MIN_ANNOTATION_SIZE,
  MIN_SAM_CLICK_REGION_SIZE,
  buildClassList,
  downloadTextFile,
  labelToColor,
  rectFromPoints,
  serializePascalVoc,
  serializeSession,
  serializeYolo,
} from './lib/annotations'
import type {
  Annotation,
  ImageEntry,
  ImageResource,
  LoadedImage,
  Point,
  Rect,
} from './types'

type ViewportTool = 'draw' | 'new-box' | 'sam-click' | 'sam-box'

const PRELOAD_RADIUS = 1
const DELETE_IMAGE_ARM_DURATION_MS = 3000
const ANNOTATION_SAVE_DEBOUNCE_MS = 350
const SIDEBAR_VISIBILITY_STORAGE_KEY = 'labelimg.sidebarVisible'
const RECENT_DATASETS_STORAGE_KEY = 'labelimg.recentDatasets'
const SESSION_STATE_STORAGE_KEY = 'labelimg.sessionState'
const HOTKEY_BINDINGS_STORAGE_KEY = 'labelimg.hotkeys'
const SAM_SETTINGS_STORAGE_KEY = 'labelimg.samSettings'
const DEFAULT_SAM_SCORE_THRESHOLD = '0.25'
const DEFAULT_SAM_MAX_RESULTS = '8'
const MAX_RECENT_DATASETS = 6
const HUGGING_FACE_OAUTH_APP_URL =
  'https://huggingface.co/settings/applications/new'
const HOTKEY_SECTION_ORDER = [
  'Session',
  'Navigation',
  'Annotation',
  'Classes',
] as const
const HOTKEY_ACTIONS = [
  {
    id: 'openImage',
    section: 'Session',
    title: 'Open image',
    description: 'Open a local image through the backend',
    defaultBindings: ['Ctrl+O'],
  },
  {
    id: 'openDataset',
    section: 'Session',
    title: 'Open dataset',
    description: 'Open a local directory through the backend',
    defaultBindings: ['Ctrl+U'],
  },
  {
    id: 'closeOverlay',
    section: 'Session',
    title: 'Close menu / dialog',
    description: 'Close the current menu, popover or lightbox',
    defaultBindings: ['Esc'],
  },
  {
    id: 'prevImage',
    section: 'Navigation',
    title: 'Previous image',
    description: 'Go to the previous image',
    defaultBindings: ['A', 'ArrowLeft'],
  },
  {
    id: 'nextImage',
    section: 'Navigation',
    title: 'Next image',
    description: 'Go to the next image',
    defaultBindings: ['D', 'ArrowRight'],
  },
  {
    id: 'deleteSelection',
    section: 'Annotation',
    title: 'Delete selected box',
    description: 'Remove the currently selected annotation',
    defaultBindings: ['Delete', 'Backspace'],
  },
  {
    id: 'startNewBox',
    section: 'Annotation',
    title: 'Start new box',
    description: 'Switch to draw mode and prepare a new box',
    defaultBindings: ['W'],
  },
  {
    id: 'autoAnnotateImage',
    section: 'Annotation',
    title: 'Auto-annotate image',
    description: 'Run SAM auto-annotation for the current image',
    defaultBindings: ['Space'],
  },
  {
    id: 'selectClass1',
    section: 'Classes',
    title: 'Select class 1',
    description: 'Activate visible class #1',
    defaultBindings: ['1'],
  },
  {
    id: 'selectClass2',
    section: 'Classes',
    title: 'Select class 2',
    description: 'Activate visible class #2',
    defaultBindings: ['2'],
  },
  {
    id: 'selectClass3',
    section: 'Classes',
    title: 'Select class 3',
    description: 'Activate visible class #3',
    defaultBindings: ['3'],
  },
  {
    id: 'selectClass4',
    section: 'Classes',
    title: 'Select class 4',
    description: 'Activate visible class #4',
    defaultBindings: ['4'],
  },
  {
    id: 'selectClass5',
    section: 'Classes',
    title: 'Select class 5',
    description: 'Activate visible class #5',
    defaultBindings: ['5'],
  },
  {
    id: 'selectClass6',
    section: 'Classes',
    title: 'Select class 6',
    description: 'Activate visible class #6',
    defaultBindings: ['6'],
  },
  {
    id: 'selectClass7',
    section: 'Classes',
    title: 'Select class 7',
    description: 'Activate visible class #7',
    defaultBindings: ['7'],
  },
  {
    id: 'selectClass8',
    section: 'Classes',
    title: 'Select class 8',
    description: 'Activate visible class #8',
    defaultBindings: ['8'],
  },
  {
    id: 'selectClass9',
    section: 'Classes',
    title: 'Select class 9',
    description: 'Activate visible class #9',
    defaultBindings: ['9'],
  },
] as const
const HOTKEY_CLASS_SLOT_ACTIONS = [
  'selectClass1',
  'selectClass2',
  'selectClass3',
  'selectClass4',
  'selectClass5',
  'selectClass6',
  'selectClass7',
  'selectClass8',
  'selectClass9',
] as const

type RecentDataset = {
  path: string
  label: string
}

type HotkeyActionId = (typeof HOTKEY_ACTIONS)[number]['id']
type HotkeyBindings = Record<HotkeyActionId, string[]>
type HotkeyCaptureTarget = {
  actionId: HotkeyActionId
  bindingIndex: number | null
}

type ConfirmDialogState = {
  title: string
  message: string
  confirmLabel: string
  confirmTone?: 'danger' | 'default'
  onConfirm: () => void
}

type PersistedSessionSourceKind = 'image' | 'dataset'

type PersistedSessionState = {
  sourceKind: PersistedSessionSourceKind
  sourcePath: string
  currentImageRelativePath: string | null
}

type ApplyLocalSessionOptions = {
  persistedState?: PersistedSessionState | null
  preferredImageRelativePath?: string | null
}

type SamPromptPairDraft = {
  prompt: string
  label: string
}

type SamSettingsDraft = {
  entries: SamPromptPairDraft[]
  scoreThreshold: string
  maxResults: string
}

type ToastTone = 'info' | 'success' | 'error'

type ToastItem = {
  id: string
  message: string
  tone: ToastTone
  isClosing: boolean
}

function App() {
  const [images, setImages] = useState<ImageEntry[]>([])
  const [imageResources, setImageResources] = useState<
    Record<string, ImageResource>
  >({})
  const [annotationsByImage, setAnnotationsByImage] = useState<
    Record<string, Annotation[]>
  >({})
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [currentImageEntry, setCurrentImageEntry] = useState<ImageEntry | null>(
    null,
  )
  const [sessionLabel, setSessionLabel] = useState('No session')
  const [isSidebarVisible, setIsSidebarVisible] = useState(
    readStoredSidebarVisibility,
  )
  const [sidebarView, setSidebarView] = useState<'main' | 'plugins'>('main')
  const [recentDatasets, setRecentDatasets] = useState(readStoredRecentDatasets)
  const [persistedSessionState, setPersistedSessionState] = useState<
    PersistedSessionState | null
  >(readStoredPersistedSessionState)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draftRect, setDraftRect] = useState<Rect | null>(null)
  const [activeLabel, setActiveLabel] = useState('object')
  const [openMenu, setOpenMenu] = useState<
    'file' | 'annotation' | 'export' | 'plugins' | 'settings' | null
  >(null)
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'online' | 'offline'
  >('checking')
  const [backendClasses, setBackendClasses] = useState<string[]>([])
  const [currentSessionRootPath, setCurrentSessionRootPath] = useState<
    string | null
  >(null)
  const [projectClassesByRootPath, setProjectClassesByRootPath] = useState<
    Record<string, string[]>
  >({})
  const [hotkeyBindings, setHotkeyBindings] = useState<HotkeyBindings>(
    readStoredHotkeyBindings,
  )
  const [hotkeyCaptureTarget, setHotkeyCaptureTarget] =
    useState<HotkeyCaptureTarget | null>(null)
  const [confirmDialogState, setConfirmDialogState] =
    useState<ConfirmDialogState | null>(null)
  const [isClassManagerOpen, setIsClassManagerOpen] = useState(false)
  const [isHotkeysOpen, setIsHotkeysOpen] = useState(false)
  const [isPluginsOpen, setIsPluginsOpen] = useState(false)
  const [newClassName, setNewClassName] = useState('')
  const [editingClassLabel, setEditingClassLabel] = useState<string | null>(null)
  const [editingClassDraft, setEditingClassDraft] = useState('')
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [pluginsError, setPluginsError] = useState<string | null>(null)
  const [selectedPluginTabId, setSelectedPluginTabId] = useState<string | null>(
    null,
  )
  const [hfAuthStatus, setHfAuthStatus] = useState<HuggingFaceAuthStatus | null>(
    null,
  )
  const [hfClientIdDraft, setHfClientIdDraft] = useState('')
  const [isHfClientIdVisible, setIsHfClientIdVisible] = useState(false)
  const [hfAuthError, setHfAuthError] = useState<string | null>(null)
  const [isStartingHfAuth, setIsStartingHfAuth] = useState(false)
  const [isSavingHfConfig, setIsSavingHfConfig] = useState(false)
  const [isCallbackCopied, setIsCallbackCopied] = useState(false)
  const [downloadingPluginId, setDownloadingPluginId] = useState<string | null>(
    null,
  )
  const [installingPluginRuntimeId, setInstallingPluginRuntimeId] = useState<
    string | null
  >(null)
  const [samPromptPairs, setSamPromptPairs] = useState<SamPromptPairDraft[]>(
    () => readStoredSamSettings().entries,
  )
  const [samScoreThresholdDraft, setSamScoreThresholdDraft] = useState(
    () => readStoredSamSettings().scoreThreshold,
  )
  const [samMaxResultsDraft, setSamMaxResultsDraft] = useState(
    () => readStoredSamSettings().maxResults,
  )
  const [samActionError, setSamActionError] = useState<string | null>(null)
  const [samActionMessage, setSamActionMessage] = useState<string | null>(null)
  const [samActionMessageTone, setSamActionMessageTone] =
    useState<ToastTone>('success')
  const [isRunningSamAction, setIsRunningSamAction] = useState(false)
  const [viewportTool, setViewportTool] = useState<ViewportTool>('draw')
  const [isOpeningSession, setIsOpeningSession] = useState(false)
  const [openingTarget, setOpeningTarget] = useState<'image' | 'dataset' | null>(
    null,
  )
  const [sessionLoadProgress, setSessionLoadProgress] = useState<{
    phase: LocalSessionJobResponse['phase']
    processed: number
    total: number
  } | null>(null)
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [armedDeleteImageId, setArmedDeleteImageId] = useState<string | null>(null)
  const [isDeletingCurrentImage, setIsDeletingCurrentImage] = useState(false)
  const [draggedAnnotationId, setDraggedAnnotationId] = useState<string | null>(
    null,
  )
  const [dragInsertIndex, setDragInsertIndex] = useState<number | null>(null)
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const menuBarRef = useRef<HTMLElement | null>(null)
  const imageResourcesRef = useRef<Record<string, ImageResource>>({})
  const annotationsByImageRef = useRef<Record<string, Annotation[]>>({})
  const imageIdSetRef = useRef<Set<string>>(new Set())
  const pendingLoadsRef = useRef<Record<string, Promise<void>>>({})
  const pendingAnnotationLoadsRef = useRef<Record<string, Promise<void>>>({})
  const annotationLoadStateRef = useRef<
    Record<string, 'idle' | 'loading' | 'ready' | 'error'>
  >({})
  const drawStartRef = useRef<Point | null>(null)
  const viewportHoverPointRef = useRef<Point | null>(null)
  const appStateSyncReadyRef = useRef(false)
  const pendingPreferredImageRelativePathRef = useRef<string | null>(null)
  const sessionVersionRef = useRef(0)
  const restoreAttemptedRef = useRef(false)
  const isMountedRef = useRef(true)
  const hfAuthPopupRef = useRef<Window | null>(null)
  const pendingHfAuthRefreshRef = useRef(false)
  const callbackCopiedTimeoutRef = useRef<number | null>(null)
  const runtimeInstallLogRef = useRef<HTMLPreElement | null>(null)
  const deleteImageArmTimeoutRef = useRef<number | null>(null)
  const annotationSaveTimeoutsRef = useRef<Record<string, number>>({})
  const annotationSaveRevisionRef = useRef<Record<string, number>>({})
  const toastTimeoutsRef = useRef<Record<string, number>>({})
  const toastRemoveTimeoutsRef = useRef<Record<string, number>>({})
  const pluginToastKeysRef = useRef<Record<string, string>>({})

  const currentImageIndex =
    currentImageEntry !== null
      ? images.findIndex((entry) => entry.id === currentImageEntry.id)
      : images.length > 0
        ? 0
        : -1
  const currentEntry =
    currentImageIndex >= 0
      ? (images[currentImageIndex] ?? null)
      : (images[0] ?? null)
  const currentResource = currentEntry
    ? imageResources[currentEntry.id] ?? null
    : null
  const image: LoadedImage | null =
    currentEntry && currentResource?.status === 'ready'
      ? {
          ...currentEntry,
          width: currentResource.width,
          height: currentResource.height,
        }
      : null
  const annotations = currentEntry
    ? annotationsByImage[currentEntry.id] ?? []
    : []
  const sessionAnnotations = Object.values(annotationsByImage).flat()
  const selectedAnnotation =
    selectedId !== null
      ? annotations.find((annotation) => annotation.id === selectedId) ?? null
      : null
  const hasStoredProjectClasses =
    currentSessionRootPath !== null
      ? Object.prototype.hasOwnProperty.call(
          projectClassesByRootPath,
          currentSessionRootPath,
        )
      : false
  const projectClasses =
    currentSessionRootPath !== null && hasStoredProjectClasses
      ? (projectClassesByRootPath[currentSessionRootPath] ?? [])
      : backendClasses
  const annotationClassList = buildClassList(sessionAnnotations)
  const classList = [...new Set([...projectClasses, ...annotationClassList])]
  const effectiveActiveLabel = classList.includes(activeLabel)
    ? activeLabel
    : (classList[0] ?? activeLabel)
  const samScoreThresholdValue = clampNumericInput(samScoreThresholdDraft, 0.25, {
    min: 0,
    max: 1,
  })
  const isDatasetSession = persistedSessionState?.sourceKind === 'dataset'
  const isDeleteImageArmed =
    currentEntry !== null && armedDeleteImageId === currentEntry.id
  const classUsageCounts = sessionAnnotations.reduce<Record<string, number>>(
    (current, annotation) => {
      current[annotation.label] = (current[annotation.label] ?? 0) + 1
      return current
    },
    {},
  )
  const hasSession = images.length > 0
  const isCurrentImageLoading =
    Boolean(currentEntry) &&
    (!currentResource || currentResource.status === 'loading')
  const isCurrentImageError = currentResource?.status === 'error'
  const currentImageSize =
    currentResource?.status === 'ready'
      ? `${currentResource.width} × ${currentResource.height}`
      : currentResource?.status === 'error'
        ? 'Failed to load'
        : currentEntry && isOpeningSession
          ? 'Opening...'
        : currentEntry
            ? 'Loading...'
            : '—'
  const openingLabel =
    openingTarget === 'dataset'
      ? sessionLoadProgress?.total
        ? `Scanning dataset, ${sessionLoadProgress.total.toLocaleString()} images indexed`
        : 'Scanning dataset'
      : 'Loading image'
  const hotkeySections = HOTKEY_SECTION_ORDER.map((sectionTitle) => ({
    title: sectionTitle,
    items: HOTKEY_ACTIONS.filter((action) => action.section === sectionTitle),
  }))
  const selectedPlugin =
    plugins.find((plugin) => plugin.id === selectedPluginTabId) ??
    plugins[0] ??
    null
  const samPlugin = plugins.find((plugin) => plugin.id === 'sam-3-1') ?? null
  const isSamInteractiveReady =
    samPlugin !== null &&
    samPlugin.model.isInstalled &&
    samPlugin.runtime.status === 'ready'
  const isAnyPluginRuntimeInstallRunning =
    installingPluginRuntimeId !== null ||
    plugins.some(
      (plugin) => getPluginRuntimeInstallState(plugin).status === 'running',
    )
  const removeToast = (toastId: string) => {
    const autoTimeoutId = toastTimeoutsRef.current[toastId]
    if (autoTimeoutId !== undefined) {
      window.clearTimeout(autoTimeoutId)
      delete toastTimeoutsRef.current[toastId]
    }

    const removeTimeoutId = toastRemoveTimeoutsRef.current[toastId]
    if (removeTimeoutId !== undefined) {
      window.clearTimeout(removeTimeoutId)
      delete toastRemoveTimeoutsRef.current[toastId]
    }

    setToasts((current) => current.filter((toast) => toast.id !== toastId))
  }
  const dismissToast = (toastId: string) => {
    const autoTimeoutId = toastTimeoutsRef.current[toastId]
    if (autoTimeoutId !== undefined) {
      window.clearTimeout(autoTimeoutId)
      delete toastTimeoutsRef.current[toastId]
    }

    setToasts((current) =>
      current.map((toast) =>
        toast.id === toastId && !toast.isClosing
          ? { ...toast, isClosing: true }
          : toast,
      ),
    )

    if (toastRemoveTimeoutsRef.current[toastId] !== undefined) {
      return
    }

    toastRemoveTimeoutsRef.current[toastId] = window.setTimeout(() => {
      removeToast(toastId)
    }, 180)
  }
  const pushToast = useEffectEvent((message: string, tone: ToastTone) => {
    const trimmedMessage = message.trim()
    if (!trimmedMessage) {
      return
    }

    const toastId = crypto.randomUUID()
    setToasts((current) => [
      ...current.slice(-4),
      {
        id: toastId,
        message: trimmedMessage,
        tone,
        isClosing: false,
      },
    ])

    toastTimeoutsRef.current[toastId] = window.setTimeout(() => {
      dismissToast(toastId)
    }, 5000)
  })
  const refreshPlugins = useEffectEvent(async (signal?: AbortSignal) => {
    try {
      const nextPlugins = await fetchPlugins(signal)
      setPlugins(nextPlugins)
      setPluginsError(null)
    } catch (error) {
      if (signal?.aborted) {
        return
      }

      setPluginsError(
        error instanceof Error ? error.message : 'Failed to load plugins',
      )
    }
  })
  const refreshHfAuthStatus = useEffectEvent(async (signal?: AbortSignal) => {
    try {
      const nextStatus = await fetchHuggingFaceAuthStatus(signal)
      setHfAuthStatus(nextStatus)
      setHfClientIdDraft(nextStatus.clientId ?? '')
      setHfAuthError(null)
    } catch (error) {
      if (signal?.aborted) {
        return
      }

      setHfAuthError(
        error instanceof Error
          ? error.message
          : 'Failed to load Hugging Face auth status',
      )
    }
  })
  const refreshAfterHfConnect = useEffectEvent(() => {
    setHfAuthError(null)
    void refreshHfAuthStatus()
    void refreshPlugins()
  })
  const matchesHotkeyAction = useEffectEvent((
    event: KeyboardEvent,
    actionId: HotkeyActionId,
  ) => {
    const normalizedBinding = normalizeHotkeyEvent(event)
    if (!normalizedBinding) {
      return false
    }

    return hotkeyBindings[actionId].includes(normalizedBinding)
  })

  const persistAppStatePatch = useEffectEvent((
    patch: {
      sidebarVisible?: boolean
      recentDatasets?: RecentDataset[]
      sessionState?: PersistedSessionState | null
      hotkeys?: HotkeyBindings
      projectClassesByRootPath?: Record<string, string[]>
      samSettings?: SamSettingsDraft
    },
  ) => {
    if (!appStateSyncReadyRef.current || backendStatus !== 'online') {
      return
    }

    void updateAppState(patch).catch(() => {
      // Ignore persistence failures and keep the current in-memory state.
    })
  })

  const commitRecentDatasets = (
    updater: (current: RecentDataset[]) => RecentDataset[],
  ) => {
    setRecentDatasets((current) => updater(current))
  }

  const updateDrawStart = (next: Point | null) => {
    drawStartRef.current = next
  }

  const handleViewportHoverPointChange = useEffectEvent((point: Point | null) => {
    viewportHoverPointRef.current = point
  })

  useEffect(() => {
    viewportHoverPointRef.current = null
  }, [currentEntry?.id])

  const commitPersistedSessionState = (next: PersistedSessionState | null) => {
    setPersistedSessionState((current) => {
      if (arePersistedSessionStatesEqual(current, next)) {
        return current
      }

      return next
    })
  }

  const clearDeleteImageArm = () => {
    if (deleteImageArmTimeoutRef.current !== null) {
      window.clearTimeout(deleteImageArmTimeoutRef.current)
      deleteImageArmTimeoutRef.current = null
    }

    setArmedDeleteImageId(null)
  }

  const updateSamPromptPair = (
    index: number,
    updater: (current: SamPromptPairDraft) => SamPromptPairDraft,
  ) => {
    setSamPromptPairs((current) =>
      current.map((entry, entryIndex) =>
        entryIndex === index ? updater(entry) : entry,
      ),
    )
  }

  const addSamPromptPair = () => {
    setSamPromptPairs((current) => [...current, createEmptySamPromptPair()])
  }

  const removeSamPromptPair = (index: number) => {
    setSamPromptPairs((current) => {
      if (current.length <= 1) {
        return [createEmptySamPromptPair()]
      }

      return current.filter((_, entryIndex) => entryIndex !== index)
    })
  }

  const updateCurrentProjectClasses = (
    updater: (current: string[]) => string[],
  ) => {
    if (!currentSessionRootPath) {
      return null
    }

    const nextProjectClasses = sanitizeClassList(updater(projectClasses))

    setProjectClassesByRootPath((current) => ({
      ...current,
      [currentSessionRootPath]: nextProjectClasses,
    }))
    commitAnnotationsByImage((current) =>
      remapProjectClassAliasesInImageMap(current, nextProjectClasses),
    )

    return nextProjectClasses
  }

  useEffect(() => {
    try {
      window.localStorage.setItem(
        SIDEBAR_VISIBILITY_STORAGE_KEY,
        isSidebarVisible ? '1' : '0',
      )
    } catch {
      // Ignore storage failures and keep the in-memory state.
    }

    persistAppStatePatch({ sidebarVisible: isSidebarVisible })
  }, [backendStatus, isSidebarVisible])

  useEffect(() => {
    try {
      window.localStorage.setItem(
        RECENT_DATASETS_STORAGE_KEY,
        JSON.stringify(recentDatasets),
      )
    } catch {
      // Ignore storage failures and keep the in-memory state.
    }

    persistAppStatePatch({ recentDatasets })
  }, [backendStatus, recentDatasets])

  useEffect(() => {
    try {
      if (persistedSessionState) {
        window.localStorage.setItem(
          SESSION_STATE_STORAGE_KEY,
          JSON.stringify(persistedSessionState),
        )
      } else {
        window.localStorage.removeItem(SESSION_STATE_STORAGE_KEY)
      }
    } catch {
      // Ignore storage failures and keep the in-memory state.
    }

    persistAppStatePatch({ sessionState: persistedSessionState })
  }, [backendStatus, persistedSessionState])

  useEffect(() => {
    try {
      window.localStorage.setItem(
        HOTKEY_BINDINGS_STORAGE_KEY,
        JSON.stringify(hotkeyBindings),
      )
    } catch {
      // Ignore storage failures and keep the in-memory state.
    }

    persistAppStatePatch({ hotkeys: hotkeyBindings })
  }, [backendStatus, hotkeyBindings])

  useEffect(() => {
    persistAppStatePatch({ projectClassesByRootPath })
  }, [backendStatus, projectClassesByRootPath])

  useEffect(() => {
    const nextSamSettings = sanitizeSamSettings({
      entries: samPromptPairs,
      scoreThreshold: samScoreThresholdDraft,
      maxResults: samMaxResultsDraft,
    })

    try {
      window.localStorage.setItem(
        SAM_SETTINGS_STORAGE_KEY,
        JSON.stringify(nextSamSettings),
      )
    } catch {
      // Ignore storage failures and keep the in-memory state.
    }

    persistAppStatePatch({ samSettings: nextSamSettings })
  }, [backendStatus, samMaxResultsDraft, samPromptPairs, samScoreThresholdDraft])

  const commitImageResources = (
    updater: (current: Record<string, ImageResource>) => Record<string, ImageResource>,
  ) => {
    setImageResources((current) => {
      const next = updater(current)
      imageResourcesRef.current = next
      return next
    })
  }

  const commitAnnotationsByImage = (
    updater: (current: Record<string, Annotation[]>) => Record<string, Annotation[]>,
  ) => {
    const next = updater(annotationsByImageRef.current)
    annotationsByImageRef.current = next
    setAnnotationsByImage(next)
    return next
  }

  const clearScheduledAnnotationSave = (imageId?: string) => {
    if (imageId) {
      const timeoutId = annotationSaveTimeoutsRef.current[imageId]
      if (timeoutId !== undefined) {
        window.clearTimeout(timeoutId)
        delete annotationSaveTimeoutsRef.current[imageId]
      }
      delete annotationSaveRevisionRef.current[imageId]
      return
    }

    for (const timeoutId of Object.values(annotationSaveTimeoutsRef.current)) {
      window.clearTimeout(timeoutId)
    }
    annotationSaveTimeoutsRef.current = {}
    annotationSaveRevisionRef.current = {}
  }

  const updateImageAnnotationMetadata = useEffectEvent((
    imageId: string,
    count: number,
    format?: string | null,
  ) => {
    setImages((current) =>
      current.map((entry) =>
        entry.id === imageId
          ? {
              ...entry,
              annotationCount: count,
              annotationFormat:
                format === undefined
                  ? entry.annotationFormat ?? null
                  : (format ?? null),
            }
          : entry,
      ),
    )
  })

  const scheduleAnnotationSave = useEffectEvent((
    imageId: string,
    nextAnnotations: Annotation[],
    projectClassesOverride?: string[],
  ) => {
    if (!currentSessionId) {
      return
    }

    const sessionId = currentSessionId
    const preferredClasses = [...(projectClassesOverride ?? projectClasses)]
    const nextRevision = (annotationSaveRevisionRef.current[imageId] ?? 0) + 1
    annotationSaveRevisionRef.current[imageId] = nextRevision

    const existingTimeoutId = annotationSaveTimeoutsRef.current[imageId]
    if (existingTimeoutId !== undefined) {
      window.clearTimeout(existingTimeoutId)
    }

    annotationSaveTimeoutsRef.current[imageId] = window.setTimeout(() => {
      delete annotationSaveTimeoutsRef.current[imageId]

      void saveLocalAnnotations(
        sessionId,
        imageId,
        nextAnnotations,
        preferredClasses,
      )
        .then((payload) => {
          if (annotationSaveRevisionRef.current[imageId] !== nextRevision) {
            return
          }

          updateImageAnnotationMetadata(
            imageId,
            payload.count,
            payload.format ?? undefined,
          )
        })
        .catch((error) => {
          if (annotationSaveRevisionRef.current[imageId] !== nextRevision) {
            return
          }

          pushToast(
            error instanceof Error
              ? error.message
              : 'Failed to save annotations',
            'error',
          )
        })
    }, ANNOTATION_SAVE_DEBOUNCE_MS)
  })

  const replaceImageAnnotations = useEffectEvent((
    imageId: string,
    nextAnnotations: Annotation[],
    options: {
      projectClassesOverride?: string[]
      selectedAnnotationId?: string | null
    } = {},
  ) => {
    annotationLoadStateRef.current[imageId] = 'ready'
    commitAnnotationsByImage((current) => ({
      ...current,
      [imageId]: nextAnnotations,
    }))
    updateImageAnnotationMetadata(imageId, nextAnnotations.length)

    if (options.selectedAnnotationId !== undefined && currentEntry?.id === imageId) {
      setSelectedId(options.selectedAnnotationId)
    }

    scheduleAnnotationSave(
      imageId,
      nextAnnotations,
      options.projectClassesOverride,
    )
  })

  const resetSessionState = (nextImages: ImageEntry[], nextSessionLabel: string) => {
    sessionVersionRef.current += 1
    imageIdSetRef.current = new Set(nextImages.map((entry) => entry.id))
    pendingLoadsRef.current = {}
    pendingAnnotationLoadsRef.current = {}
    annotationLoadStateRef.current = {}
    imageResourcesRef.current = {}
    clearScheduledAnnotationSave()
    pendingPreferredImageRelativePathRef.current = null

    startTransition(() => {
      setImageResources({})
      setImages(nextImages)
      commitAnnotationsByImage(() => ({}))
      setCurrentSessionId(null)
      setCurrentSessionRootPath(null)
      setCurrentImageEntry(nextImages[0] ?? null)
      setSessionLabel(nextSessionLabel)
      setSessionError(null)
      setSelectedId(null)
      updateDrawStart(null)
      setDraftRect(null)
      setIsClassManagerOpen(false)
      setNewClassName('')
      setEditingClassLabel(null)
      setEditingClassDraft('')
    })
  }

  const ensureImageResource = useEffectEvent(async (entry: ImageEntry) => {
    const existingResource = imageResourcesRef.current[entry.id]
    if (existingResource?.status === 'ready' || existingResource?.status === 'loading') {
      return
    }

    const pendingLoad = pendingLoadsRef.current[entry.id]
    if (pendingLoad) {
      return pendingLoad
    }

    commitImageResources((current) => ({
      ...current,
      [entry.id]: {
        width: current[entry.id]?.width ?? 0,
        height: current[entry.id]?.height ?? 0,
        status: 'loading',
      },
    }))

    const sessionVersion = sessionVersionRef.current
    const promise = preloadImage(entry.url)
      .then((size) => {
        if (
          sessionVersion !== sessionVersionRef.current ||
          !imageIdSetRef.current.has(entry.id)
        ) {
          return
        }

        commitImageResources((current) => {
          return {
            ...current,
            [entry.id]: {
              width: size.width,
              height: size.height,
              status: 'ready',
            },
          }
        })
      })
      .catch(() => {
        if (
          sessionVersion !== sessionVersionRef.current ||
          !imageIdSetRef.current.has(entry.id)
        ) {
          return
        }

        commitImageResources((current) => ({
          ...current,
          [entry.id]: {
            width: current[entry.id]?.width ?? 0,
            height: current[entry.id]?.height ?? 0,
            status: 'error',
          },
        }))
      })
      .finally(() => {
        if (pendingLoadsRef.current[entry.id] === promise) {
          delete pendingLoadsRef.current[entry.id]
        }
      })

    pendingLoadsRef.current[entry.id] = promise
    return promise
  })

  useEffect(() => {
    return () => {
      isMountedRef.current = false
      sessionVersionRef.current += 1
      pendingLoadsRef.current = {}
      pendingAnnotationLoadsRef.current = {}
      annotationLoadStateRef.current = {}
      clearScheduledAnnotationSave()
    }
  }, [])

  useEffect(() => {
    return () => {
      for (const timeoutId of Object.values(toastTimeoutsRef.current)) {
        window.clearTimeout(timeoutId)
      }
      for (const timeoutId of Object.values(toastRemoveTimeoutsRef.current)) {
        window.clearTimeout(timeoutId)
      }
      toastTimeoutsRef.current = {}
      toastRemoveTimeoutsRef.current = {}
    }
  }, [])

  useEffect(() => {
    return () => {
      if (deleteImageArmTimeoutRef.current !== null) {
        window.clearTimeout(deleteImageArmTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!armedDeleteImageId) {
      return
    }

    if (!currentEntry || currentEntry.id !== armedDeleteImageId) {
      clearDeleteImageArm()
      return
    }

    deleteImageArmTimeoutRef.current = window.setTimeout(() => {
      deleteImageArmTimeoutRef.current = null
      setArmedDeleteImageId((current) =>
        current === armedDeleteImageId ? null : current,
      )
    }, DELETE_IMAGE_ARM_DURATION_MS)

    return () => {
      if (deleteImageArmTimeoutRef.current !== null) {
        window.clearTimeout(deleteImageArmTimeoutRef.current)
        deleteImageArmTimeoutRef.current = null
      }
    }
  }, [armedDeleteImageId, currentEntry])

  const ensureAnnotationsLoaded = useEffectEvent(async (entry: ImageEntry) => {
    if (!currentSessionId) {
      return
    }

    const currentLoadState = annotationLoadStateRef.current[entry.id]
    if (
      currentLoadState === 'loading' ||
      currentLoadState === 'ready' ||
      currentLoadState === 'error'
    ) {
      return
    }

    const pendingLoad = pendingAnnotationLoadsRef.current[entry.id]
    if (pendingLoad) {
      return pendingLoad
    }

    annotationLoadStateRef.current[entry.id] = 'loading'
    const sessionVersion = sessionVersionRef.current
    const promise = fetchLocalAnnotations(currentSessionId, entry.id)
      .then((payload) => {
        if (
          sessionVersion !== sessionVersionRef.current ||
          !imageIdSetRef.current.has(entry.id)
        ) {
          return
        }

        const nextAnnotations = payload.annotations.map((annotation) => {
          const label = resolveProjectClassAlias(annotation.label, projectClasses)
          return {
            ...annotation,
            label,
            color: labelToColor(label),
          }
        })

        setImages((current) =>
          current.map((imageEntry) =>
            imageEntry.id === entry.id
              ? {
                  ...imageEntry,
                  annotationCount: payload.count ?? nextAnnotations.length,
                }
              : imageEntry,
          ),
        )

        commitAnnotationsByImage((current) => {
          if (current[entry.id] !== undefined) {
            return current
          }

          return {
            ...current,
            [entry.id]: nextAnnotations,
          }
        })

        annotationLoadStateRef.current[entry.id] = 'ready'
      })
      .catch(() => {
        if (
          sessionVersion !== sessionVersionRef.current ||
          !imageIdSetRef.current.has(entry.id)
        ) {
          return
        }

        annotationLoadStateRef.current[entry.id] = 'error'
      })
      .finally(() => {
        if (pendingAnnotationLoadsRef.current[entry.id] === promise) {
          delete pendingAnnotationLoadsRef.current[entry.id]
        }
      })

    pendingAnnotationLoadsRef.current[entry.id] = promise
    return promise
  })

  useEffect(() => {
    const controller = new AbortController()

    const bootstrapApi = async () => {
      try {
        await fetchApiHealth(controller.signal)
        const [classes, appState, pluginList, hfAuth] = await Promise.all([
          fetchPredefinedClasses(controller.signal),
          fetchAppState(controller.signal).catch(() => null),
          fetchPlugins(controller.signal).catch(() => null),
          fetchHuggingFaceAuthStatus(controller.signal).catch(() => null),
        ])
        if (controller.signal.aborted) {
          return
        }

        if (appState) {
          const nextSidebarVisibility = coerceSidebarVisibility(
            appState.sidebarVisible,
          )
          if (nextSidebarVisibility !== null) {
            setIsSidebarVisible(nextSidebarVisibility)
          }

          setRecentDatasets(coerceRecentDatasets(appState.recentDatasets))
          setPersistedSessionState(
            coercePersistedSessionState(appState.sessionState),
          )
          setHotkeyBindings(coerceHotkeyBindings(appState.hotkeys))
          setProjectClassesByRootPath(
            coerceProjectClassesByRootPath(appState.projectClassesByRootPath),
          )
          if (appState.samSettings !== undefined) {
            const nextSamSettings = coerceSamSettings(appState.samSettings)
            setSamPromptPairs(nextSamSettings.entries)
            setSamScoreThresholdDraft(nextSamSettings.scoreThreshold)
            setSamMaxResultsDraft(nextSamSettings.maxResults)
          }
        }

        setBackendClasses(classes)
        setPlugins(pluginList ?? [])
        setPluginsError(pluginList === null ? 'Failed to load plugins' : null)
        setHfAuthStatus(hfAuth)
        setHfClientIdDraft(hfAuth?.clientId ?? '')
        setHfAuthError(hfAuth === null ? 'Failed to load Hugging Face auth status' : null)
        setActiveLabel((current) =>
          current === 'object' && classes.length > 0 ? classes[0] : current,
        )
        appStateSyncReadyRef.current = true
        setBackendStatus('online')
      } catch {
        if (controller.signal.aborted) {
          return
        }

        appStateSyncReadyRef.current = false
        setBackendStatus('offline')
      }
    }

    void bootstrapApi()

    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (
      backendStatus !== 'online' ||
      restoreAttemptedRef.current ||
      !persistedSessionState ||
      hasSession ||
      isOpeningSession
    ) {
      return
    }

    let isCancelled = false
    restoreAttemptedRef.current = true
    setSessionError(null)
    setOpeningTarget(persistedSessionState.sourceKind)
    setSessionLoadProgress(
      persistedSessionState.sourceKind === 'dataset'
        ? {
            phase: 'indexing',
            processed: 0,
            total: 0,
          }
        : null,
    )
    setIsOpeningSession(true)

    const restoreSession = async () => {
      try {
        if (persistedSessionState.sourceKind === 'dataset') {
          const job = await openLocalDirectoryPathJob(
            persistedSessionState.sourcePath,
          )
          if (!job.cancelled && job.jobId) {
            await waitForSessionJob(
              job.jobId,
              setSessionLoadProgress,
              (session) => {
                applyLocalSession(session, {
                  persistedState: persistedSessionState,
                  preferredImageRelativePath:
                    persistedSessionState.currentImageRelativePath,
                })
              },
              rememberRecentDataset,
            )
          }
          return
        }

        const session = await openLocalImagePath(persistedSessionState.sourcePath)

        if (isCancelled || session.cancelled) {
          return
        }

        applyLocalSession(session, {
          persistedState: persistedSessionState,
          preferredImageRelativePath:
            persistedSessionState.currentImageRelativePath,
        })
      } catch (error) {
        if (isCancelled) {
          return
        }

        commitPersistedSessionState(null)
        setSessionError(
          error instanceof Error
            ? error.message
            : 'Failed to restore previous session',
        )
      } finally {
        if (isMountedRef.current) {
          setIsOpeningSession(false)
          setOpeningTarget(null)
          setSessionLoadProgress(null)
        }
      }
    }

    void restoreSession()

    return () => {
      isCancelled = true
    }
  }, [backendStatus, hasSession, isOpeningSession, persistedSessionState])

  useEffect(() => {
    if (!sessionError) {
      return
    }

    pushToast(sessionError, 'error')
  }, [sessionError])

  useEffect(() => {
    if (!pluginsError) {
      return
    }

    pushToast(pluginsError, 'error')
  }, [pluginsError])

  useEffect(() => {
    if (!hfAuthError) {
      return
    }

    pushToast(hfAuthError, 'error')
  }, [hfAuthError])

  useEffect(() => {
    if (!samActionError) {
      return
    }

    pushToast(samActionError, 'error')
  }, [samActionError])

  useEffect(() => {
    if (!samActionMessage) {
      return
    }

    pushToast(samActionMessage, samActionMessageTone)
  }, [samActionMessage, samActionMessageTone])

  useEffect(() => {
    if (!isPluginsOpen && sidebarView !== 'plugins') {
      return
    }

    const nextToastKeys = { ...pluginToastKeysRef.current }
    const maybePushPluginToast = (
      notificationId: string,
      key: string,
      message: string,
      tone: ToastTone,
    ) => {
      if (nextToastKeys[notificationId] === key) {
        return
      }

      nextToastKeys[notificationId] = key
      pushToast(message, tone)
    }

    for (const plugin of plugins) {
      const download = getPluginDownloadState(plugin)
      if (download.status === 'failed' && download.error) {
        maybePushPluginToast(
          `${plugin.id}:download`,
          `${download.status}:${download.error}:${download.finishedAt ?? download.startedAt ?? ''}`,
          download.error,
          'error',
        )
      }

      const runtimeInstall = getPluginRuntimeInstallState(plugin)
      if (runtimeInstall.status === 'completed' && runtimeInstall.message) {
        maybePushPluginToast(
          `${plugin.id}:runtime-install`,
          `${runtimeInstall.status}:${runtimeInstall.message}:${runtimeInstall.finishedAt ?? runtimeInstall.startedAt ?? ''}`,
          runtimeInstall.message,
          'success',
        )
      } else if (runtimeInstall.status === 'failed' && runtimeInstall.error) {
        maybePushPluginToast(
          `${plugin.id}:runtime-install`,
          `${runtimeInstall.status}:${runtimeInstall.error}:${runtimeInstall.finishedAt ?? runtimeInstall.startedAt ?? ''}`,
          runtimeInstall.error,
          'error',
        )
      } else if (
        runtimeInstall.status === 'idle' &&
        plugin.runtime.status !== 'ready' &&
        plugin.runtime.message
      ) {
        maybePushPluginToast(
          `${plugin.id}:runtime-status`,
          `${plugin.runtime.status}:${plugin.runtime.message}`,
          plugin.runtime.message,
          plugin.runtime.status === 'error' ? 'error' : 'info',
        )
      }
    }

    pluginToastKeysRef.current = nextToastKeys
  }, [isPluginsOpen, plugins, pushToast, sidebarView])

  useEffect(() => {
    if (!openMenu) {
      return
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (
        menuBarRef.current &&
        event.target instanceof Node &&
        !menuBarRef.current.contains(event.target)
      ) {
        setOpenMenu(null)
      }
    }

    const handleCloseHotkey = (event: KeyboardEvent) => {
      if (matchesHotkeyAction(event, 'closeOverlay')) {
        event.preventDefault()
        setOpenMenu(null)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleCloseHotkey)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleCloseHotkey)
    }
  }, [openMenu])

  useEffect(() => {
    if (!isClassManagerOpen) {
      return
    }

    const handleCloseHotkey = (event: KeyboardEvent) => {
      if (matchesHotkeyAction(event, 'closeOverlay')) {
        event.preventDefault()
        closeClassManager()
      }
    }

    window.addEventListener('keydown', handleCloseHotkey)
    return () => window.removeEventListener('keydown', handleCloseHotkey)
  }, [isClassManagerOpen])

  useEffect(() => {
    if (!isHotkeysOpen) {
      return
    }

    const handleCloseHotkey = (event: KeyboardEvent) => {
      if (matchesHotkeyAction(event, 'closeOverlay')) {
        event.preventDefault()
        closeHotkeys()
      }
    }

    window.addEventListener('keydown', handleCloseHotkey)
    return () => window.removeEventListener('keydown', handleCloseHotkey)
  }, [isHotkeysOpen])

  useEffect(() => {
    if (!isPluginsOpen) {
      return
    }

    const handleCloseHotkey = (event: KeyboardEvent) => {
      if (matchesHotkeyAction(event, 'closeOverlay')) {
        event.preventDefault()
        closePluginsManager()
      }
    }

    window.addEventListener('keydown', handleCloseHotkey)
    return () => window.removeEventListener('keydown', handleCloseHotkey)
  }, [isPluginsOpen])

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) {
        return
      }

      const payload = event.data as
        | { type?: string; success?: boolean; message?: string }
        | null
        | undefined
      if (payload?.type !== 'hf-oauth-complete') {
        return
      }

      hfAuthPopupRef.current?.close()
      hfAuthPopupRef.current = null
      pendingHfAuthRefreshRef.current = false
      setIsStartingHfAuth(false)

      if (payload.success) {
        refreshAfterHfConnect()
      } else {
        setHfAuthError(payload.message ?? 'Hugging Face authorization failed')
      }
    }

    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [])

  useEffect(() => {
    if (!isStartingHfAuth) {
      return
    }

    const intervalId = window.setInterval(() => {
      const popup = hfAuthPopupRef.current
      if (!popup || popup.closed) {
        hfAuthPopupRef.current = null
        if (pendingHfAuthRefreshRef.current) {
          pendingHfAuthRefreshRef.current = false
          refreshAfterHfConnect()
        }
        setIsStartingHfAuth(false)
        window.clearInterval(intervalId)
      }
    }, 350)

    return () => window.clearInterval(intervalId)
  }, [isStartingHfAuth])

  useEffect(() => {
    return () => {
      if (callbackCopiedTimeoutRef.current !== null) {
        window.clearTimeout(callbackCopiedTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (plugins.length === 0) {
      if (selectedPluginTabId !== null) {
        setSelectedPluginTabId(null)
      }
      return
    }

    if (!selectedPluginTabId || !plugins.some((plugin) => plugin.id === selectedPluginTabId)) {
      setSelectedPluginTabId(plugins[0]?.id ?? null)
    }
  }, [plugins, selectedPluginTabId])

  useEffect(() => {
    if (sidebarView !== 'plugins') {
      return
    }

    const controller = new AbortController()
    void refreshPlugins(controller.signal)
    void refreshHfAuthStatus(controller.signal)
    return () => controller.abort()
  }, [sidebarView])

  useEffect(() => {
    if (!downloadingPluginId) {
      return
    }

    const intervalId = window.setInterval(() => {
      void refreshPlugins()
    }, 500)

    return () => window.clearInterval(intervalId)
  }, [downloadingPluginId])

  useEffect(() => {
    if (!downloadingPluginId) {
      return
    }

    const activePlugin = plugins.find((plugin) => plugin.id === downloadingPluginId)
    if (!activePlugin) {
      return
    }
    const activeDownload = getPluginDownloadState(activePlugin)

    if (activeDownload.status === 'failed') {
      setPluginsError(activeDownload.error || 'Failed to download plugin model')
      setDownloadingPluginId(null)
      return
    }

    if (activeDownload.status === 'completed' || activePlugin.model.isInstalled) {
      setDownloadingPluginId(null)
    }
  }, [plugins, downloadingPluginId])

  useEffect(() => {
    if (!installingPluginRuntimeId) {
      return
    }

    const intervalId = window.setInterval(() => {
      void refreshPlugins()
    }, 1000)

    return () => window.clearInterval(intervalId)
  }, [installingPluginRuntimeId])

  useEffect(() => {
    if (!installingPluginRuntimeId) {
      return
    }

    const activePlugin =
      plugins.find((plugin) => plugin.id === installingPluginRuntimeId) ?? null
    if (!activePlugin) {
      return
    }

    const activeInstall = getPluginRuntimeInstallState(activePlugin)
    if (activeInstall.status === 'failed') {
      setPluginsError(activeInstall.error || 'Failed to install plugin runtime')
      setInstallingPluginRuntimeId(null)
      return
    }

    if (activeInstall.status === 'completed') {
      setInstallingPluginRuntimeId(null)
    }
  }, [plugins, installingPluginRuntimeId])

  useEffect(() => {
    if (!installingPluginRuntimeId) {
      return
    }

    const activePlugin =
      plugins.find((plugin) => plugin.id === installingPluginRuntimeId) ?? null
    if (!activePlugin) {
      return
    }

    const activeInstall = getPluginRuntimeInstallState(activePlugin)
    if (!activeInstall.log) {
      return
    }

    const frameId = window.requestAnimationFrame(() => {
      const terminalNode = runtimeInstallLogRef.current
      if (!terminalNode) {
        return
      }
      terminalNode.scrollTop = terminalNode.scrollHeight
    })

    return () => window.cancelAnimationFrame(frameId)
  }, [plugins, installingPluginRuntimeId])

  useEffect(() => {
    if (!confirmDialogState) {
      return
    }

    const handleCloseHotkey = (event: KeyboardEvent) => {
      if (matchesHotkeyAction(event, 'closeOverlay')) {
        event.preventDefault()
        setConfirmDialogState(null)
      }
    }

    window.addEventListener('keydown', handleCloseHotkey)
    return () => window.removeEventListener('keydown', handleCloseHotkey)
  }, [confirmDialogState])

  useEffect(() => {
    if (
      !isSamInteractiveReady &&
      (viewportTool === 'sam-click' || viewportTool === 'sam-box')
    ) {
      setViewportTool('draw')
    }
  }, [isSamInteractiveReady, viewportTool])

  useEffect(() => {
    if (!hotkeyCaptureTarget) {
      return
    }

    const handleCapture = (event: KeyboardEvent) => {
      event.preventDefault()
      event.stopPropagation()

      if (event.repeat) {
        return
      }

      const binding = normalizeHotkeyEvent(event)
      if (!binding) {
        return
      }

      setHotkeyBindings((current) =>
        applyHotkeyCapture(current, hotkeyCaptureTarget, binding),
      )
      setHotkeyCaptureTarget(null)
    }

    window.addEventListener('keydown', handleCapture, true)
    return () => window.removeEventListener('keydown', handleCapture, true)
  }, [hotkeyCaptureTarget])

  useEffect(() => {
    if (!currentEntry) {
      return
    }

    if (currentImageIndex < 0) {
      return
    }

    for (const entry of getBufferedEntries(images, currentImageIndex, PRELOAD_RADIUS)) {
      void ensureImageResource(entry)
      void ensureAnnotationsLoaded(entry)
    }
  }, [currentEntry, currentImageIndex, images])

  useEffect(() => {
    if (!currentEntry || !currentSessionId || !persistedSessionState) {
      return
    }

    if (
      persistedSessionState.currentImageRelativePath === currentEntry.relativePath
    ) {
      return
    }

    commitPersistedSessionState({
      ...persistedSessionState,
      currentImageRelativePath: currentEntry.relativePath,
    })
  }, [currentEntry, currentSessionId, persistedSessionState])

  const selectImageIndex = (nextIndex: number) => {
    if (images.length === 0) {
      return
    }

    const boundedIndex = Math.min(Math.max(nextIndex, 0), images.length - 1)
    const nextEntry = images[boundedIndex] ?? null
    if (!nextEntry) {
      return
    }

    pendingPreferredImageRelativePathRef.current = null
    setSelectedId(null)
    updateDrawStart(null)
    setDraftRect(null)
    setCurrentImageEntry(nextEntry)
  }

  const goPrevImage = () => {
    if (currentImageIndex <= 0) {
      return
    }

    const nextEntry = images[currentImageIndex - 1] ?? null
    if (!nextEntry) {
      return
    }

    pendingPreferredImageRelativePathRef.current = null
    setSelectedId(null)
    updateDrawStart(null)
    setDraftRect(null)
    setCurrentImageEntry(nextEntry)
  }

  const goNextImage = () => {
    if (currentImageIndex < 0 || currentImageIndex >= images.length - 1) {
      return
    }

    const nextEntry = images[currentImageIndex + 1] ?? null
    if (!nextEntry) {
      return
    }

    pendingPreferredImageRelativePathRef.current = null
    setSelectedId(null)
    updateDrawStart(null)
    setDraftRect(null)
    setCurrentImageEntry(nextEntry)
  }

  const clearSession = () => {
    setOpenMenu(null)
    commitPersistedSessionState(null)
    resetSessionState([], 'No session')
  }

  const openClassManager = () => {
    if (!currentSessionRootPath) {
      return
    }

    setOpenMenu(null)
    setIsClassManagerOpen(true)
    setNewClassName('')
    setEditingClassLabel(null)
    setEditingClassDraft('')
  }

  const closeClassManager = () => {
    setIsClassManagerOpen(false)
    setNewClassName('')
    setEditingClassLabel(null)
    setEditingClassDraft('')
  }

  const openHotkeys = () => {
    setOpenMenu(null)
    setIsHotkeysOpen(true)
  }

  const closeHotkeys = () => {
    setHotkeyCaptureTarget(null)
    setIsHotkeysOpen(false)
  }

  const selectSidebarView = (nextView: 'main' | 'plugins') => {
    setSidebarView(nextView)
    if (nextView === 'plugins') {
      setPluginsError(null)
      setHfAuthError(null)
      void refreshPlugins()
      void refreshHfAuthStatus()
    }
  }

  const openPluginsManager = () => {
    setOpenMenu(null)
    setPluginsError(null)
    setHfAuthError(null)
    setIsPluginsOpen(true)
    void refreshPlugins()
    void refreshHfAuthStatus()
  }

  const closePluginsManager = () => {
    setIsPluginsOpen(false)
    setDownloadingPluginId(null)
  }

  const handleDownloadPlugin = useEffectEvent(async (pluginId: string) => {
    setPluginsError(null)
    setDownloadingPluginId(pluginId)

    try {
      const updatedPlugin = await downloadPluginModel(pluginId)
      setPlugins((current) =>
        current.map((plugin) =>
          plugin.id === updatedPlugin.id ? updatedPlugin : plugin,
        ),
      )
      if (updatedPlugin.download.status !== 'running') {
        setDownloadingPluginId(null)
      }
    } catch (error) {
      setPluginsError(
        error instanceof Error ? error.message : 'Failed to download plugin model',
      )
      void refreshPlugins()
    } finally {
      setDownloadingPluginId((current) =>
        current === pluginId ? null : current,
      )
    }
  })

  const handleInstallPluginRuntime = useEffectEvent(
    async (pluginId: string, profile: PluginRuntimeInstallProfile) => {
      setPluginsError(null)
      setInstallingPluginRuntimeId(pluginId)

      try {
        const updatedPlugin = await installPluginRuntime(pluginId, profile)
        setPlugins((current) =>
          current.map((plugin) =>
            plugin.id === updatedPlugin.id ? updatedPlugin : plugin,
          ),
        )

        if (getPluginRuntimeInstallState(updatedPlugin).status !== 'running') {
          setInstallingPluginRuntimeId(null)
        }
      } catch (error) {
        setPluginsError(
          error instanceof Error ? error.message : 'Failed to install plugin runtime',
        )
        void refreshPlugins()
      }
    },
  )

  const runSamAutoAnnotate = useEffectEvent(
    async ({
      mode,
      region = null,
      replaceAnnotationId = null,
      successMessage,
      emptyMessage = 'SAM found no matching objects for this prompt.',
      requirePrompt = true,
    }: {
      mode: 'full-image' | 'selected-box'
      region?: Rect | null
      replaceAnnotationId?: string | null
      successMessage?: string
      emptyMessage?: string
      requirePrompt?: boolean
    }) => {
      if (!samPlugin) {
        setSamActionError('SAM 3.1 plugin is not registered.')
        return
      }

      if (!currentEntry) {
        setSamActionError('Open an image or dataset before using SAM auto-annotation.')
        return
      }

      if (mode === 'selected-box' && !region) {
        setSamActionError('Select a box or draw a region before running box-guided SAM.')
        return
      }

      const fallbackTerm = effectiveActiveLabel.trim() || 'object'
      const configuredPairs = samPromptPairs
        .map((entry) => {
          const prompt = entry.prompt.trim()
          const label = entry.label.trim()

          if (!prompt && !label) {
            return null
          }

          return {
            prompt: prompt || label || fallbackTerm,
            label: label || fallbackTerm || prompt || 'object',
          }
        })
        .filter(
          (
            entry,
          ): entry is {
            prompt: string
            label: string
          } => entry !== null,
        )
      const activePairs =
        mode === 'selected-box' ? configuredPairs.slice(0, 1) : configuredPairs
      const pairsToRun =
        activePairs.length > 0 || requirePrompt
          ? activePairs
          : [{ prompt: fallbackTerm, label: fallbackTerm }]

      if (pairsToRun.length === 0) {
        setSamActionError(
          mode === 'selected-box'
            ? 'Add at least one Prompt / Label pair before refining with SAM.'
            : 'Add at least one Prompt / Label pair before running SAM auto-annotation.',
        )
        return
      }

      setSamActionError(null)
      setSamActionMessage(null)
      setIsRunningSamAction(true)

      const scoreThreshold = clampNumericInput(samScoreThresholdDraft, 0.25, {
        min: 0,
        max: 1,
      })
      const maxResults = Math.round(
        clampNumericInput(samMaxResultsDraft, 8, {
          min: 1,
          max: 64,
        }),
      )

      try {
        const emptyResultMessage =
          pairsToRun.length > 1
            ? 'SAM found no matching objects for the configured prompt list.'
            : emptyMessage

        if (replaceAnnotationId) {
          const primaryPair = pairsToRun[0]!
          const result = await runPluginAutoAnnotate('sam-3-1', {
            sessionId: currentSessionId,
            imageId: currentEntry.id,
            prompt: primaryPair.prompt,
            label: primaryPair.label,
            mode,
            region,
            scoreThreshold,
            maxResults,
          })

          if (result.annotations.length === 0) {
            setSamActionMessageTone('info')
            setSamActionMessage(emptyResultMessage)
            return
          }

          const bestMatch = result.annotations[0]!
          const currentAnnotations =
            annotationsByImageRef.current[currentEntry.id] ?? []
          replaceImageAnnotations(
            currentEntry.id,
            currentAnnotations.map((annotation) =>
              annotation.id === replaceAnnotationId
                ? {
                    ...annotation,
                    label: bestMatch.label,
                    color: labelToColor(bestMatch.label),
                    x: bestMatch.x,
                    y: bestMatch.y,
                    width: bestMatch.width,
                    height: bestMatch.height,
                  }
                : annotation,
            ),
            { selectedAnnotationId: replaceAnnotationId },
          )
          setSamActionMessageTone('success')
          setSamActionMessage(successMessage ?? 'SAM refined the selected box.')
          return
        }

        const collectedAnnotations: PluginAutoAnnotateResult['annotations'] = []

        for (const pair of pairsToRun) {
          const result = await runPluginAutoAnnotate('sam-3-1', {
            sessionId: currentSessionId,
            imageId: currentEntry.id,
            prompt: pair.prompt,
            label: pair.label,
            mode,
            region,
            scoreThreshold,
            maxResults,
          })

          if (result.annotations.length > 0) {
            collectedAnnotations.push(...result.annotations)
          }
        }

        if (collectedAnnotations.length === 0) {
          setSamActionMessageTone('info')
          setSamActionMessage(emptyResultMessage)
          return
        }

        const nextAnnotations = collectedAnnotations.map((annotation) => ({
          id: crypto.randomUUID(),
          label: annotation.label,
          color: labelToColor(annotation.label),
          difficult: false,
          x: annotation.x,
          y: annotation.y,
          width: annotation.width,
          height: annotation.height,
        }))

        const currentAnnotations =
          annotationsByImageRef.current[currentEntry.id] ?? []
        replaceImageAnnotations(
          currentEntry.id,
          [...currentAnnotations, ...nextAnnotations],
          { selectedAnnotationId: nextAnnotations[0]?.id ?? null },
        )
        setSamActionMessageTone('success')
        setSamActionMessage(
          successMessage ??
            (pairsToRun.length > 1
              ? `SAM added ${nextAnnotations.length} auto-annotation${nextAnnotations.length === 1 ? '' : 's'} from ${pairsToRun.length} prompt/label pair${pairsToRun.length === 1 ? '' : 's'}.`
              : `SAM added ${nextAnnotations.length} auto-annotation${nextAnnotations.length === 1 ? '' : 's'}.`),
        )
      } catch (error) {
        setSamActionError(
          error instanceof Error ? error.message : 'SAM auto-annotation failed',
        )
        void refreshPlugins()
      } finally {
        setIsRunningSamAction(false)
      }
    },
  )

  const handleRunSamAutoAnnotate = useEffectEvent(
    async (mode: 'full-image' | 'selected-box') => {
      if (mode === 'selected-box') {
        if (!selectedAnnotation) {
          setSamActionError('Select a box before running box-guided SAM refinement.')
          return
        }

        await runSamAutoAnnotate({
          mode,
          region: {
            x: selectedAnnotation.x,
            y: selectedAnnotation.y,
            width: selectedAnnotation.width,
            height: selectedAnnotation.height,
          },
          replaceAnnotationId: selectedAnnotation.id,
        })
        return
      }

      await runSamAutoAnnotate({
        mode,
        requirePrompt: true,
      })
    },
  )

  const handleStartHuggingFaceAuth = useEffectEvent(async () => {
    setHfAuthError(null)
    setIsStartingHfAuth(true)
    pendingHfAuthRefreshRef.current = true

    const popup = window.open(
      '',
      'huggingface-oauth',
      'popup,width=640,height=760,resizable=yes,scrollbars=yes',
    )
    hfAuthPopupRef.current = popup

    try {
      const response = await startHuggingFaceAuth()
      if (popup) {
        popup.location.href = response.authorizationUrl
        popup.focus()
      } else {
        window.location.href = response.authorizationUrl
      }
    } catch (error) {
      popup?.close()
      hfAuthPopupRef.current = null
      pendingHfAuthRefreshRef.current = false
      setHfAuthError(
        error instanceof Error ? error.message : 'Failed to start Hugging Face login',
      )
      setIsStartingHfAuth(false)
    }
  })

  const handleCopyHfCallbackUrl = useEffectEvent(async () => {
    const callbackUrl = hfAuthStatus?.callbackUrl?.trim()
    if (!callbackUrl) {
      return
    }

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(callbackUrl)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = callbackUrl
        textarea.setAttribute('readonly', 'true')
        textarea.style.position = 'absolute'
        textarea.style.left = '-9999px'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        document.body.removeChild(textarea)
      }
      setIsCallbackCopied(true)
      if (callbackCopiedTimeoutRef.current !== null) {
        window.clearTimeout(callbackCopiedTimeoutRef.current)
      }
      callbackCopiedTimeoutRef.current = window.setTimeout(() => {
        setIsCallbackCopied(false)
        callbackCopiedTimeoutRef.current = null
      }, 1800)
    } catch (error) {
      setHfAuthError(
        error instanceof Error ? error.message : 'Failed to copy callback URL',
      )
    }
  })

  const handleSaveHuggingFaceAuthConfig = useEffectEvent(async () => {
    setHfAuthError(null)
    setIsSavingHfConfig(true)

    try {
      const nextStatus = await updateHuggingFaceAuthConfig(
        hfClientIdDraft.trim() || null,
      )
      setHfAuthStatus(nextStatus)
      setHfClientIdDraft(nextStatus.clientId ?? '')
      setIsHfClientIdVisible(false)
    } catch (error) {
      setHfAuthError(
        error instanceof Error
          ? error.message
          : 'Failed to save Hugging Face OAuth config',
      )
    } finally {
      setIsSavingHfConfig(false)
    }
  })

  const handleClearHuggingFaceAuthConfig = useEffectEvent(async () => {
    setHfAuthError(null)
    setIsSavingHfConfig(true)

    try {
      const nextStatus = await updateHuggingFaceAuthConfig(null)
      setHfAuthStatus(nextStatus)
      setHfClientIdDraft('')
      setIsHfClientIdVisible(false)
    } catch (error) {
      setHfAuthError(
        error instanceof Error
          ? error.message
          : 'Failed to clear Hugging Face OAuth config',
      )
    } finally {
      setIsSavingHfConfig(false)
    }
  })

  const handleLogoutHuggingFaceAuth = useEffectEvent(async () => {
    setHfAuthError(null)

    try {
      const nextStatus = await logoutHuggingFaceAuth()
      setHfAuthStatus(nextStatus)
      void refreshPlugins()
    } catch (error) {
      setHfAuthError(
        error instanceof Error ? error.message : 'Failed to disconnect Hugging Face',
      )
    }
  })

  const beginHotkeyCapture = (
    actionId: HotkeyActionId,
    bindingIndex: number | null,
  ) => {
    setHotkeyCaptureTarget({ actionId, bindingIndex })
  }

  const cancelHotkeyCapture = () => {
    setHotkeyCaptureTarget(null)
  }

  const closeConfirmDialog = () => {
    setConfirmDialogState(null)
  }

  const openResetHotkeysConfirmDialog = () => {
    setConfirmDialogState({
      title: 'Reset hotkeys to defaults?',
      message:
        'All custom keyboard shortcuts will be replaced with the default bindings.',
      confirmLabel: 'Reset hotkeys',
      confirmTone: 'danger',
      onConfirm: () => {
        setHotkeyBindings(buildDefaultHotkeyBindings())
        setHotkeyCaptureTarget(null)
        setConfirmDialogState(null)
      },
    })
  }

  const removeHotkeyBinding = (
    actionId: HotkeyActionId,
    bindingIndex: number,
  ) => {
    setHotkeyBindings((current) => ({
      ...current,
      [actionId]: current[actionId].filter((_, index) => index !== bindingIndex),
    }))
  }

  const resetHotkeysToDefaults = () => {
    openResetHotkeysConfirmDialog()
  }

  const addClass = () => {
    const nextLabel = newClassName.trim()
    if (!nextLabel) {
      return
    }

    if (!classList.includes(nextLabel)) {
      updateCurrentProjectClasses((current) => [...current, nextLabel])
    }

    setActiveLabel(nextLabel)
    setNewClassName('')
  }

  const startEditingClass = (label: string) => {
    setEditingClassLabel(label)
    setEditingClassDraft(label)
  }

  const cancelEditingClass = () => {
    setEditingClassLabel(null)
    setEditingClassDraft('')
  }

  const renameClass = (sourceLabel: string) => {
    const nextLabel = editingClassDraft.trim()
    if (!nextLabel) {
      return
    }

    if (nextLabel === sourceLabel) {
      cancelEditingClass()
      return
    }

    const hasConflict = classList.some(
      (label) => label === nextLabel && label !== sourceLabel,
    )
    if (hasConflict) {
      return
    }

    const nextProjectClasses = updateCurrentProjectClasses((current) => {
      const next = current.filter(
        (label) => label !== sourceLabel && label !== nextLabel,
      )
      next.push(nextLabel)
      return next
    })

    const changedImageIds: string[] = []
    const nextAnnotationsByImage = commitAnnotationsByImage((current) =>
      Object.fromEntries(
        Object.entries(current).map(([imageId, imageAnnotations]) => [
          imageId,
          imageAnnotations.map((annotation) => {
            if (annotation.label !== sourceLabel) {
              return annotation
            }

            if (!changedImageIds.includes(imageId)) {
              changedImageIds.push(imageId)
            }

            return {
              ...annotation,
              label: nextLabel,
              color: labelToColor(nextLabel),
            }
          }),
        ]),
      ),
    )

    for (const imageId of changedImageIds) {
      annotationLoadStateRef.current[imageId] = 'ready'
      updateImageAnnotationMetadata(
        imageId,
        nextAnnotationsByImage[imageId]?.length ?? 0,
      )
      scheduleAnnotationSave(
        imageId,
        nextAnnotationsByImage[imageId] ?? [],
        nextProjectClasses ?? undefined,
      )
    }

    setActiveLabel((current) =>
      current === sourceLabel ? nextLabel : current,
    )
    cancelEditingClass()
  }

  const removeClass = (label: string) => {
    if ((classUsageCounts[label] ?? 0) > 0) {
      return
    }

    updateCurrentProjectClasses((current) =>
      current.filter((entry) => entry !== label),
    )
    setActiveLabel((current) =>
      current === label ? classList.find((entry) => entry !== label) ?? 'object' : current,
    )
    if (editingClassLabel === label) {
      cancelEditingClass()
    }
  }

  const removeAnnotation = useEffectEvent((annotationId: string) => {
    if (!currentEntry) {
      return
    }

    replaceImageAnnotations(
      currentEntry.id,
      annotations.filter((annotation) => annotation.id !== annotationId),
      {
        selectedAnnotationId: selectedId === annotationId ? null : undefined,
      },
    )
  })

  const duplicateAnnotation = useEffectEvent((annotationId: string) => {
    if (!currentEntry) {
      return
    }

    const sourceAnnotation = annotations.find(
      (annotation) => annotation.id === annotationId,
    )
    if (!sourceAnnotation) {
      return
    }

    const offset = 12
    const maxX = image ? Math.max(0, image.width - sourceAnnotation.width) : null
    const maxY = image ? Math.max(0, image.height - sourceAnnotation.height) : null
    const nextAnnotation: Annotation = {
      ...sourceAnnotation,
      id: crypto.randomUUID(),
      x:
        maxX === null
          ? sourceAnnotation.x + offset
          : Math.min(Math.max(sourceAnnotation.x + offset, 0), maxX),
      y:
        maxY === null
          ? sourceAnnotation.y + offset
          : Math.min(Math.max(sourceAnnotation.y + offset, 0), maxY),
    }

    replaceImageAnnotations(
      currentEntry.id,
      [...annotations, nextAnnotation],
      { selectedAnnotationId: nextAnnotation.id },
    )
  })

  const deleteSelectedAnnotation = () => {
    if (selectedId) {
      removeAnnotation(selectedId)
    }
  }

  const updateAnnotationRect = useEffectEvent((
    annotationId: string,
    nextRect: Rect,
  ) => {
    if (!currentEntry) {
      return
    }

    replaceImageAnnotations(
      currentEntry.id,
      annotations.map((annotation) =>
        annotation.id === annotationId
          ? {
              ...annotation,
              ...nextRect,
            }
          : annotation,
      ),
    )
  })

  const changeAnnotationLabel = useEffectEvent((
    annotationId: string,
    nextLabel: string,
  ) => {
    if (!currentEntry) {
      return
    }

    const trimmedLabel = nextLabel.trim()
    if (!trimmedLabel) {
      return
    }

    replaceImageAnnotations(
      currentEntry.id,
      annotations.map((annotation) =>
        annotation.id === annotationId
          ? {
              ...annotation,
              label: trimmedLabel,
              color: labelToColor(trimmedLabel),
            }
          : annotation,
      ),
      { selectedAnnotationId: annotationId },
    )
    setActiveLabel(trimmedLabel)
  })

  const moveCurrentImageAnnotation = (
    annotationId: string,
    insertIndex: number,
  ) => {
    if (!currentEntry) {
      return
    }

    const sourceIndex = annotations.findIndex(
      (annotation) => annotation.id === annotationId,
    )
    if (sourceIndex < 0) {
      return
    }

    const boundedInsertIndex = Math.max(0, Math.min(insertIndex, annotations.length))
    if (
      boundedInsertIndex === sourceIndex ||
      boundedInsertIndex === sourceIndex + 1
    ) {
      return
    }

    const nextAnnotations = [...annotations]
    const [movedAnnotation] = nextAnnotations.splice(sourceIndex, 1)
    const adjustedInsertIndex =
      sourceIndex < boundedInsertIndex
        ? boundedInsertIndex - 1
        : boundedInsertIndex
    nextAnnotations.splice(adjustedInsertIndex, 0, movedAnnotation)

    replaceImageAnnotations(currentEntry.id, nextAnnotations)
  }

  const clearCurrentImageAnnotations = () => {
    if (!currentEntry) {
      return
    }

    replaceImageAnnotations(currentEntry.id, [], { selectedAnnotationId: null })
  }

  const handleDeleteCurrentImage = useEffectEvent(async () => {
    if (
      !currentSessionId ||
      !currentEntry ||
      !currentSessionRootPath ||
      !isDatasetSession
    ) {
      return
    }

    if (armedDeleteImageId !== currentEntry.id) {
      clearDeleteImageArm()
      setArmedDeleteImageId(currentEntry.id)
      return
    }

    const imageToDelete = currentEntry
    const preferredImageRelativePath =
      images[currentImageIndex + 1]?.relativePath ??
      images[currentImageIndex - 1]?.relativePath ??
      null

    clearDeleteImageArm()
    setIsDeletingCurrentImage(true)
    setSessionError(null)

    try {
      const session = await deleteLocalSessionImage(
        currentSessionId,
        imageToDelete.id,
      )

      applyLocalSession(session, {
        preferredImageRelativePath,
      })

      if (!preferredImageRelativePath && isDatasetSession && persistedSessionState) {
        commitPersistedSessionState({
          ...persistedSessionState,
          currentImageRelativePath: null,
        })
      }

      pushToast(`Deleted ${imageToDelete.relativePath}.`, 'success')
    } catch (error) {
      setSessionError(
        error instanceof Error ? error.message : 'Failed to delete image',
      )
    } finally {
      setIsDeletingCurrentImage(false)
    }
  })

  const startNewBoxDrawing = useEffectEvent(() => {
    if (!currentEntry) {
      return
    }

    if (!image) {
      pushToast('Wait until the image finishes loading to start a new box.', 'info')
      return
    }

    const startPoint =
      viewportHoverPointRef.current ?? {
        x: image.width / 2,
        y: image.height / 2,
      }

    setViewportTool('new-box')
    setSelectedId(null)
    updateDrawStart(startPoint)
    setDraftRect({ x: startPoint.x, y: startPoint.y, width: 0, height: 0 })
  })

  const handleSelectViewportTool = useEffectEvent((tool: ViewportTool) => {
    setViewportTool(tool)
    if (tool !== 'new-box') {
      updateDrawStart(null)
      setDraftRect(null)
    }
  })

  const onGlobalKeyDown = useEffectEvent((event: KeyboardEvent) => {
    if (
      hotkeyCaptureTarget ||
      confirmDialogState ||
      isClassManagerOpen ||
      isHotkeysOpen ||
      isPluginsOpen
    ) {
      return
    }

    if (isEditableTarget(event.target)) {
      return
    }

    if (matchesHotkeyAction(event, 'openImage')) {
      event.preventDefault()
      void handleOpenLocalImage()
      return
    }

    if (matchesHotkeyAction(event, 'openDataset')) {
      event.preventDefault()
      void handleOpenLocalDirectory()
      return
    }

    for (let index = 0; index < HOTKEY_CLASS_SLOT_ACTIONS.length; index += 1) {
      if (!matchesHotkeyAction(event, HOTKEY_CLASS_SLOT_ACTIONS[index])) {
        continue
      }

      const shortcutLabel = classList[index]
      if (!shortcutLabel) {
        return
      }

      event.preventDefault()
      setActiveLabel(shortcutLabel)
      return
    }

    if (matchesHotkeyAction(event, 'deleteSelection')) {
      event.preventDefault()
      deleteSelectedAnnotation()
      return
    }

    const matchesStartNewBoxDefaultKey =
      !event.ctrlKey &&
      !event.altKey &&
      !event.metaKey &&
      !event.shiftKey &&
      (event.code === 'KeyW' || event.key.toLowerCase() === 'w')

    if (matchesHotkeyAction(event, 'startNewBox') || matchesStartNewBoxDefaultKey) {
      if (event.repeat) {
        return
      }

      event.preventDefault()
      startNewBoxDrawing()
      return
    }

    if (matchesHotkeyAction(event, 'autoAnnotateImage')) {
      event.preventDefault()
      void handleRunSamAutoAnnotate('full-image')
      return
    }

    if (matchesHotkeyAction(event, 'prevImage')) {
      event.preventDefault()
      goPrevImage()
      return
    }

    if (matchesHotkeyAction(event, 'nextImage')) {
      event.preventDefault()
      goNextImage()
    }
  })

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      onGlobalKeyDown(event)
    }

    document.addEventListener('keydown', onKeyDown, true)
    return () => document.removeEventListener('keydown', onKeyDown, true)
  }, [])

  const applyLocalSession = useEffectEvent((
    session: LocalSessionResponse,
    options: ApplyLocalSessionOptions = {},
  ) => {
    const sessionId = session.sessionId
    const sessionImages = session.images
    const nextSessionLabel = session.sessionLabel

    if (!sessionId || !sessionImages || !nextSessionLabel) {
      return
    }

    const nextImages = sessionImages.map((entry) => ({
      ...entry,
      annotationCount: entry.annotationCount ?? 0,
      annotationFormat: entry.annotationFormat ?? null,
      url: buildLocalImageUrl(sessionId, entry.id),
    }))
    const nextImageIdSet = new Set(nextImages.map((entry) => entry.id))
    const preferredImageRelativePath =
      options.preferredImageRelativePath ??
      pendingPreferredImageRelativePathRef.current
    const firstAnnotatedIndex = nextImages.findIndex(
      (entry) => entry.annotationCount > 0,
    )
    const preferredImageIndex = preferredImageRelativePath
      ? nextImages.findIndex(
          (entry) =>
            entry.relativePath === preferredImageRelativePath,
        )
      : -1
    const initialImageIndex =
      preferredImageIndex >= 0
        ? preferredImageIndex
        : firstAnnotatedIndex >= 0
          ? firstAnnotatedIndex
          : 0
    const initialEntry = nextImages[initialImageIndex] ?? nextImages[0] ?? null

    if (currentSessionId === sessionId) {
      const removedImageIds = [...imageIdSetRef.current].filter(
        (imageId) => !nextImageIdSet.has(imageId),
      )

      for (const imageId of removedImageIds) {
        clearScheduledAnnotationSave(imageId)
      }

      imageIdSetRef.current = nextImageIdSet
      setCurrentSessionRootPath(session.rootPath ?? null)

      if (
        preferredImageRelativePath &&
        preferredImageIndex >= 0
      ) {
        if (
          pendingPreferredImageRelativePathRef.current === preferredImageRelativePath
        ) {
          pendingPreferredImageRelativePathRef.current = null
        }
        setCurrentImageEntry(nextImages[preferredImageIndex] ?? null)
      }

      annotationLoadStateRef.current = Object.fromEntries(
        Object.entries(annotationLoadStateRef.current).filter(([imageId]) =>
          nextImageIdSet.has(imageId),
        ),
      )
      imageResourcesRef.current = Object.fromEntries(
        Object.entries(imageResourcesRef.current).filter(([imageId]) =>
          nextImageIdSet.has(imageId),
        ),
      )

      startTransition(() => {
        setImages((current) => {
          const currentById = new Map(
            current.map((entry) => [entry.id, entry] as const),
          )

          return nextImages.map((entry) => ({
            ...(currentById.get(entry.id) ?? {}),
            ...entry,
          }))
        })
        setCurrentImageEntry((current) => {
          if (!current) {
            return nextImages[0] ?? null
          }

          return (
            nextImages.find((entry) => entry.id === current.id) ??
            nextImages[0] ??
            null
          )
        })
        setImageResources((current) =>
          Object.fromEntries(
            Object.entries(current).filter(([imageId]) => nextImageIdSet.has(imageId)),
          ),
        )
        commitAnnotationsByImage((current) =>
          Object.fromEntries(
            Object.entries(current).filter(([imageId]) => nextImageIdSet.has(imageId)),
          ),
        )
        setSessionLabel(nextSessionLabel)
        setSessionError(null)
      })
      return
    }

    sessionVersionRef.current += 1
    imageIdSetRef.current = nextImageIdSet
    pendingLoadsRef.current = {}
    pendingAnnotationLoadsRef.current = {}
    annotationLoadStateRef.current = {}
    imageResourcesRef.current = {}
    clearScheduledAnnotationSave()
    pendingPreferredImageRelativePathRef.current =
      preferredImageRelativePath && preferredImageIndex < 0
        ? preferredImageRelativePath
        : null

    if (options.persistedState) {
      commitPersistedSessionState({
        ...options.persistedState,
        currentImageRelativePath: initialEntry?.relativePath ?? null,
      })
    }

    startTransition(() => {
      setImageResources({})
      setImages(nextImages)
      commitAnnotationsByImage(() => ({}))
      setCurrentSessionId(sessionId)
      setCurrentSessionRootPath(session.rootPath ?? null)
      setCurrentImageEntry(initialEntry)
      setSessionLabel(nextSessionLabel)
      setSessionLoadProgress(null)
      setSessionError(null)
      setSelectedId(null)
      updateDrawStart(null)
      setDraftRect(null)
      setIsClassManagerOpen(false)
      setNewClassName('')
      setEditingClassLabel(null)
      setEditingClassDraft('')
    })
  })

  const rememberRecentDataset = (session: LocalSessionResponse) => {
    if (!session.rootPath) {
      return
    }

    const label =
      session.sessionLabel?.trim() || labelFromDatasetPath(session.rootPath)

    commitRecentDatasets((current) =>
      [
        { path: session.rootPath!, label },
        ...current.filter((entry) => entry.path !== session.rootPath),
      ].slice(0, MAX_RECENT_DATASETS),
    )
  }

  const removeRecentDataset = useEffectEvent((path: string) => {
    commitRecentDatasets((current) =>
      current.filter((entry) => entry.path !== path),
    )
  })

  const handleOpenLocalImage = async () => {
    setOpenMenu(null)
    setSessionError(null)
    setOpeningTarget('image')
    setSessionLoadProgress(null)
    setIsOpeningSession(true)

    try {
      const session = await openLocalImage()
      if (!session.cancelled) {
        applyLocalSession(session, {
          persistedState: buildPersistedSessionState(session, 'image'),
        })
      }
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : 'Failed to open image')
    } finally {
      setIsOpeningSession(false)
      setOpeningTarget(null)
    }
  }

  const handleOpenLocalDirectory = useEffectEvent(async () => {
    setOpenMenu(null)
    setSessionError(null)
    setOpeningTarget('dataset')
    setSessionLoadProgress({
      phase: 'indexing',
      processed: 0,
      total: 0,
    })
    setIsOpeningSession(true)

    try {
      const job = await openLocalDirectory()
      if (!job.cancelled && job.jobId) {
        await waitForSessionJob(
          job.jobId,
          setSessionLoadProgress,
          (session) => {
            applyLocalSession(session, {
              persistedState: buildPersistedSessionState(session, 'dataset'),
            })
          },
          rememberRecentDataset,
        )
      }
    } catch (error) {
      setSessionError(
        error instanceof Error ? error.message : 'Failed to open directory',
      )
    } finally {
      setIsOpeningSession(false)
      setOpeningTarget(null)
      setSessionLoadProgress(null)
    }
  })

  const handleOpenRecentDataset = useEffectEvent(async (path: string) => {
    setSessionError(null)
    setOpeningTarget('dataset')
    setSessionLoadProgress({
      phase: 'indexing',
      processed: 0,
      total: 0,
    })
    setIsOpeningSession(true)

    try {
      const job = await openLocalDirectoryPathJob(path)
      if (!job.cancelled && job.jobId) {
        await waitForSessionJob(
          job.jobId,
          setSessionLoadProgress,
          (session) => {
            applyLocalSession(session, {
              persistedState: buildPersistedSessionState(session, 'dataset'),
            })
          },
          rememberRecentDataset,
        )
      }
    } catch (error) {
      setSessionError(
        error instanceof Error ? error.message : 'Failed to open directory',
      )
    } finally {
      setIsOpeningSession(false)
      setOpeningTarget(null)
      setSessionLoadProgress(null)
    }
  })

  const handleCanvasPointerDown = useEffectEvent((point: Point) => {
    if (viewportTool === 'sam-click') {
      if (!image || !isSamInteractiveReady || isRunningSamAction) {
        return
      }

      const clickRegionSize = Math.max(
        MIN_SAM_CLICK_REGION_SIZE,
        Math.round(Math.min(image.width, image.height) * 0.04),
      )
      const halfSize = clickRegionSize / 2
      const region = {
        x: Math.max(0, point.x - halfSize),
        y: Math.max(0, point.y - halfSize),
        width: Math.min(clickRegionSize, image.width),
        height: Math.min(clickRegionSize, image.height),
      }

      void runSamAutoAnnotate({
        mode: 'selected-box',
        region,
        successMessage: 'SAM selected an object near the click.',
        emptyMessage: 'SAM could not find a matching object near this click.',
        requirePrompt: false,
      })
      return
    }

    setSelectedId(null)
    updateDrawStart(point)
    setDraftRect({ x: point.x, y: point.y, width: 0, height: 0 })
  })

  const handleCanvasPointerMove = useEffectEvent((point: Point) => {
    const currentDrawStart = drawStartRef.current
    if (!currentDrawStart) {
      return
    }

    const nextRect = rectFromPoints(currentDrawStart, point)
    setDraftRect(nextRect)
  })

  const handleCanvasPointerUp = useEffectEvent((point: Point) => {
    const currentDrawStart = drawStartRef.current
    if (!currentDrawStart || !image) {
      return
    }

    const nextRect = rectFromPoints(currentDrawStart, point)
    const shouldResetNewBoxMode = viewportTool === 'new-box'

    if (
      nextRect.width < MIN_ANNOTATION_SIZE ||
      nextRect.height < MIN_ANNOTATION_SIZE
    ) {
      if (!shouldResetNewBoxMode) {
        updateDrawStart(null)
        setDraftRect(null)
      }

      return
    }

    updateDrawStart(null)
    setDraftRect(null)

    if (viewportTool === 'sam-box') {
      void runSamAutoAnnotate({
        mode: 'selected-box',
        region: nextRect,
        successMessage: 'SAM selected an object inside the drawn region.',
        emptyMessage: 'SAM could not find a matching object inside this region.',
        requirePrompt: false,
      })
      return
    }

    const label = effectiveActiveLabel.trim() || 'object'
    const nextAnnotation: Annotation = {
      id: crypto.randomUUID(),
      label,
      color: labelToColor(label),
      difficult: false,
      ...nextRect,
    }

    replaceImageAnnotations(
      image.id,
      [...annotations, nextAnnotation],
      { selectedAnnotationId: nextAnnotation.id },
    )

    if (shouldResetNewBoxMode) {
      setViewportTool('draw')
    }
  })

  const handleLabelExport = (format: 'json' | 'voc' | 'yolo') => {
    if (!image) {
      return
    }

    if (format !== 'yolo' && annotations.length === 0) {
      return
    }

    const baseName = stripExtension(image.name) || 'annotation'

    if (format === 'json') {
      downloadTextFile(
        `${baseName}.labelimg-next.json`,
        serializeSession(image, annotations),
        'application/json;charset=utf-8',
      )
      return
    }

    if (format === 'voc') {
      downloadTextFile(
        `${baseName}.xml`,
        serializePascalVoc(image, annotations),
        'application/xml;charset=utf-8',
      )
      return
    }

    const yolo = serializeYolo(image, annotations, projectClasses)
    downloadTextFile(`${baseName}.txt`, yolo.annotationText)
    downloadTextFile('classes.txt', yolo.classesText)
  }

  return (
    <div className="app-shell">
      {toasts.length > 0 ? (
        <div className="toast-stack" aria-live="polite" aria-label="Notifications">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className={[
                'toast-notification',
                `is-${toast.tone}`,
                toast.isClosing ? 'is-closing' : '',
              ]
                .filter(Boolean)
                .join(' ')}
              role={toast.tone === 'error' ? 'alert' : 'status'}
            >
              <div className="toast-notification-message">{toast.message}</div>
              <button
                type="button"
                className="toast-notification-close"
                onClick={() => dismissToast(toast.id)}
                aria-label="Close notification"
                title="Close notification"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <header className="menubar" ref={menuBarRef}>
        <nav className="menubar-nav" aria-label="Main menu">
          <div className="menu-root">
            <AppButton
              variant="menu-trigger"
              isActive={openMenu === 'file'}
              onClick={() =>
                setOpenMenu((current) => (current === 'file' ? null : 'file'))
              }
              disabled={isOpeningSession}
            >
              File
            </AppButton>
            {openMenu === 'file' ? (
              <div className="menu-popover" role="menu" aria-label="File">
                <MenuItemButton
                  title="Open Image"
                  description="Open a local image through the Python backend"
                  shortcut="Ctrl+O"
                  onClick={() => void handleOpenLocalImage()}
                  disabled={isOpeningSession || backendStatus !== 'online'}
                />
                <MenuItemButton
                  title="Open Directory"
                  description="Open a local folder without browser upload"
                  shortcut="Ctrl+U"
                  onClick={() => void handleOpenLocalDirectory()}
                  disabled={isOpeningSession || backendStatus !== 'online'}
                />
                <MenuItemButton
                  title="Close Session"
                  description="Clear the current dataset from the browser"
                  onClick={clearSession}
                  disabled={!hasSession}
                />
              </div>
            ) : null}
          </div>

          <div className="menu-root">
            <AppButton
              variant="menu-trigger"
              isActive={openMenu === 'annotation'}
              onClick={() =>
                setOpenMenu((current) => (current === 'annotation' ? null : 'annotation'))
              }
            >
              Annotation
            </AppButton>
            {openMenu === 'annotation' ? (
              <div className="menu-popover" role="menu" aria-label="Annotation">
                <MenuItemButton
                  title="Delete Selected Box"
                  description="Remove the currently selected annotation"
                  shortcut="Delete"
                  onClick={() => {
                    deleteSelectedAnnotation()
                    setOpenMenu(null)
                  }}
                  disabled={!selectedId}
                />
                <MenuItemButton
                  title="Clear Current Image"
                  description="Remove all boxes from the current image"
                  onClick={() => {
                    clearCurrentImageAnnotations()
                    setOpenMenu(null)
                  }}
                  disabled={!currentEntry || annotations.length === 0}
                />
              </div>
            ) : null}
          </div>

          <div className="menu-root">
            <AppButton
              variant="menu-trigger"
              isActive={openMenu === 'export'}
              onClick={() =>
                setOpenMenu((current) => (current === 'export' ? null : 'export'))
              }
            >
              Export
            </AppButton>
            {openMenu === 'export' ? (
              <div className="menu-popover" role="menu" aria-label="Export">
                <MenuItemButton
                  title="Export JSON"
                  description="Save the current image labels as session JSON"
                  onClick={() => {
                    handleLabelExport('json')
                    setOpenMenu(null)
                  }}
                  disabled={!image || annotations.length === 0}
                />
                <MenuItemButton
                  title="Export Pascal VOC"
                  description="Save the current image labels as VOC XML"
                  onClick={() => {
                    handleLabelExport('voc')
                    setOpenMenu(null)
                  }}
                  disabled={!image || annotations.length === 0}
                />
                <MenuItemButton
                  title="Export YOLO"
                  description="Save the current image labels as YOLO TXT"
                  onClick={() => {
                    handleLabelExport('yolo')
                    setOpenMenu(null)
                  }}
                  disabled={!image}
                />
              </div>
            ) : null}
          </div>

          <div className="menu-root">
            <AppButton
              variant="menu-trigger"
              isActive={openMenu === 'plugins'}
              onClick={() =>
                setOpenMenu((current) => (current === 'plugins' ? null : 'plugins'))
              }
            >
              Plugins
            </AppButton>
            {openMenu === 'plugins' ? (
              <div className="menu-popover" role="menu" aria-label="Plugins">
                <MenuItemButton
                  title="Manage Plugins"
                  description="Install plugin models and review their status"
                  onClick={openPluginsManager}
                  disabled={backendStatus !== 'online'}
                />
              </div>
            ) : null}
          </div>

          <div className="menu-root">
            <AppButton
              variant="menu-trigger"
              isActive={openMenu === 'settings'}
              onClick={() =>
                setOpenMenu((current) =>
                  current === 'settings' ? null : 'settings',
                )
              }
            >
              Settings
            </AppButton>
            {openMenu === 'settings' ? (
              <div className="menu-popover" role="menu" aria-label="Settings">
                <MenuItemButton
                  title="Project Classes"
                  description="Edit the YOLO classes used for the current project"
                  onClick={openClassManager}
                  disabled={!hasSession}
                />
                <MenuItemButton
                  title="Hotkeys"
                  description="View and edit keyboard shortcuts"
                  onClick={openHotkeys}
                />
              </div>
            ) : null}
          </div>

        </nav>

      </header>

      <div
        className={
          isSidebarVisible ? 'workspace' : 'workspace workspace-sidebar-hidden'
        }
      >
        <main className="annotator">
          <div className="canvas-panel">
            <AnnotationViewport
              key={currentEntry?.id ?? 'empty'}
              image={image}
              imageLabel={currentEntry?.relativePath ?? null}
              isLoading={isCurrentImageLoading}
              isError={isCurrentImageError}
              annotations={annotations}
              selectedId={selectedId}
              draftRect={draftRect}
              tool={viewportTool}
              showSamTools={isSamInteractiveReady}
              isSamBusy={isRunningSamAction}
              onSelectTool={handleSelectViewportTool}
              onOpenDataset={handleOpenLocalDirectory}
              recentDatasets={recentDatasets}
              onOpenRecentDataset={handleOpenRecentDataset}
              onRemoveRecentDataset={removeRecentDataset}
              openDatasetDisabled={isOpeningSession || backendStatus !== 'online'}
              onStartDrawing={handleCanvasPointerDown}
              onUpdateDrawing={handleCanvasPointerMove}
              onFinishDrawing={handleCanvasPointerUp}
              classOptions={classList}
              onSelectAnnotation={setSelectedId}
              onUpdateAnnotationRect={updateAnnotationRect}
              onChangeAnnotationLabel={changeAnnotationLabel}
              onDuplicateAnnotation={duplicateAnnotation}
              onDeleteAnnotation={removeAnnotation}
              onHoverPointChange={handleViewportHoverPointChange}
            />
          </div>
        </main>

        <AppButton
          variant="menu-trigger"
          className="workspace-sidebar-toggle"
          onClick={() => setIsSidebarVisible((current) => !current)}
          aria-label={isSidebarVisible ? 'Hide sidebar' : 'Show sidebar'}
          title={isSidebarVisible ? 'Hide sidebar' : 'Show sidebar'}
        >
          {isSidebarVisible ? 'Close Menu' : 'Open Menu'}
        </AppButton>

        {isSidebarVisible ? (
          <aside className="panel panel-sidebar">
            <div className="panel-sidebar-tabs" role="tablist" aria-label="Sidebar views">
              <AppButton
                variant="chip"
                isActive={sidebarView === 'main'}
                className="panel-sidebar-tab"
                onClick={() => selectSidebarView('main')}
              >
                Main Menu
              </AppButton>
              <AppButton
                variant="chip"
                isActive={sidebarView === 'plugins'}
                className="panel-sidebar-tab"
                onClick={() => selectSidebarView('plugins')}
              >
                Plugins
              </AppButton>
            </div>

            {sidebarView === 'main' ? (
              <>
                <section className="panel-section">
                  <h2>Dataset</h2>
                  <div className="meta-list">
                    <div className="meta-line meta-line-value-only">
                      <strong className="meta-value" title={sessionLabel}>
                        {sessionLabel}
                      </strong>
                    </div>
                    <div className="meta-line">
                      <span>Progress</span>
                      <div className="dataset-progress-meta" aria-live="polite">
                        <strong className="meta-value">
                          {images.length > 0
                            ? `${currentImageIndex + 1}/${images.length}`
                            : '0/0'}
                        </strong>
                        {isOpeningSession ? (
                          <span
                            className="dataset-progress-spinner"
                            role="status"
                            aria-label={openingLabel}
                            title={openingLabel}
                          >
                            <span className="visually-hidden">{openingLabel}</span>
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <div className="meta-line">
                      <span>Size</span>
                      <strong className="meta-value">{currentImageSize}</strong>
                    </div>
                  </div>
                  {images.length > 0 ? (
                    <VirtualFileList
                      images={images}
                      currentIndex={currentImageIndex}
                      onSelectIndex={selectImageIndex}
                    />
                  ) : null}
                  {isDatasetSession && currentEntry ? (
                    <AppButton
                      variant="menu-trigger"
                      className={[
                        'dataset-delete-image-button',
                        isDeleteImageArmed ? 'is-armed' : '',
                        isDeletingCurrentImage ? 'is-working' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      onClick={() => void handleDeleteCurrentImage()}
                      disabled={isDeletingCurrentImage || isOpeningSession}
                      title={
                        isDeletingCurrentImage
                          ? `Deleting ${currentEntry.relativePath}`
                          : isDeleteImageArmed
                          ? `Delete ${currentEntry.relativePath}`
                          : `Arm delete for ${currentEntry.relativePath}`
                      }
                    >
                      <span className="dataset-delete-image-copy">
                        <span className="dataset-delete-image-title">
                          {isDeletingCurrentImage
                            ? 'Deleting image...'
                            : isDeleteImageArmed
                              ? 'Click again to delete'
                              : 'Delete current image'}
                        </span>
                        <span className="dataset-delete-image-subtitle">
                          {isDeleteImageArmed
                            ? 'Armed for 3 seconds'
                            : 'Two-step delete confirmation'}
                        </span>
                      </span>
                      <span className="dataset-delete-image-indicator" aria-hidden="true">
                        {isDeleteImageArmed ? (
                          <svg viewBox="0 0 24 24" className="dataset-delete-image-ring">
                            <circle cx="12" cy="12" r="9" />
                            <circle
                              cx="12"
                              cy="12"
                              r="9"
                              className="dataset-delete-image-ring-progress"
                            />
                          </svg>
                        ) : (
                          <span className="dataset-delete-image-dot" />
                        )}
                      </span>
                    </AppButton>
                  ) : null}
                </section>

                <section className="panel-section">
                  <div className="section-heading-row section-heading-row-kicker">
                    <p className="section-kicker section-kicker-inline">Labeling</p>
                    <AppButton
                      variant="ghost"
                      className="class-add-button"
                      onClick={openClassManager}
                      disabled={!hasSession}
                    >
                      Project classes
                    </AppButton>
                  </div>
                  <AppButton
                    variant="primary"
                    isActive={viewportTool === 'new-box'}
                    className="labeling-new-box-button"
                    onClick={() => startNewBoxDrawing()}
                    disabled={!image}
                    title={
                      viewportTool === 'new-box'
                        ? 'Move the mouse and click to place the new box'
                        : 'Start a new box (W)'
                    }
                  >
                    {viewportTool === 'new-box' ? 'Click to place' : 'New box (W)'}
                  </AppButton>
                  <div className="chip-list" aria-label="Known classes">
                    {classList.length > 0 ? (
                      classList.map((label) => (
                        <AppButton
                          key={label}
                          variant="chip"
                          isActive={label === effectiveActiveLabel}
                          onClick={() => setActiveLabel(label)}
                        >
                          {label}
                        </AppButton>
                      ))
                    ) : (
                      <span className="muted">
                        Add project classes in Settings or create annotations.
                      </span>
                    )}
                  </div>
                </section>

                <section className="panel-section">
                  <p className="section-kicker">Current image</p>
                  {annotations.length > 0 ? (
                    <div
                      className="current-box-list"
                      aria-label="Current image box labels"
                      onDragOver={(event) => {
                        if (
                          !draggedAnnotationId ||
                          event.target !== event.currentTarget
                        ) {
                          return
                        }

                        event.preventDefault()
                        if (dragInsertIndex !== annotations.length) {
                          setDragInsertIndex(annotations.length)
                        }
                      }}
                      onDrop={(event) => {
                        if (
                          !draggedAnnotationId ||
                          dragInsertIndex === null ||
                          event.target !== event.currentTarget
                        ) {
                          return
                        }

                        event.preventDefault()
                        moveCurrentImageAnnotation(draggedAnnotationId, dragInsertIndex)
                        setDraggedAnnotationId(null)
                        setDragInsertIndex(null)
                      }}
                    >
                      {annotations.map((annotation, index) => (
                        <div
                          key={annotation.id}
                          className={[
                            'current-box-row',
                            draggedAnnotationId === annotation.id
                              ? 'is-dragging'
                              : '',
                            draggedAnnotationId !== annotation.id &&
                            dragInsertIndex === index
                              ? 'drop-before'
                              : '',
                            draggedAnnotationId !== annotation.id &&
                            dragInsertIndex === index + 1
                              ? 'drop-after'
                              : '',
                          ]
                            .filter(Boolean)
                            .join(' ')}
                          draggable
                          onDragStart={(event) => {
                            event.dataTransfer.effectAllowed = 'move'
                            event.dataTransfer.setData('text/plain', annotation.id)
                            setSelectedId(annotation.id)
                            setDraggedAnnotationId(annotation.id)
                            setDragInsertIndex(index)
                          }}
                          onDragOver={(event) => {
                            if (!draggedAnnotationId) {
                              return
                            }

                            event.preventDefault()
                            const bounds = event.currentTarget.getBoundingClientRect()
                            const nextInsertIndex =
                              event.clientY < bounds.top + bounds.height / 2
                                ? index
                                : index + 1

                            if (dragInsertIndex !== nextInsertIndex) {
                              setDragInsertIndex(nextInsertIndex)
                            }
                          }}
                          onDrop={(event) => {
                            if (!draggedAnnotationId || dragInsertIndex === null) {
                              return
                            }

                            event.preventDefault()
                            moveCurrentImageAnnotation(draggedAnnotationId, dragInsertIndex)
                            setDraggedAnnotationId(null)
                            setDragInsertIndex(null)
                          }}
                          onDragEnd={() => {
                            setDraggedAnnotationId(null)
                            setDragInsertIndex(null)
                          }}
                        >
                          <AppButton
                            variant="list-row"
                            isActive={annotation.id === selectedId}
                            className="current-box-item"
                            onClick={() => setSelectedId(annotation.id)}
                            title={annotation.label || 'object'}
                          >
                            <span className="current-box-index">{index + 1}</span>
                            <span
                              className="current-box-swatch"
                              style={{ backgroundColor: annotation.color }}
                              aria-hidden="true"
                            />
                            <span className="current-box-name">
                              {annotation.label || 'object'}
                            </span>
                          </AppButton>
                          <div className="current-box-actions">
                            <AppButton
                              variant="ghost"
                              className="current-box-action"
                              title="Duplicate box"
                              aria-label={`Duplicate ${annotation.label || 'object'}`}
                              onClick={(event) => {
                                event.stopPropagation()
                                duplicateAnnotation(annotation.id)
                              }}
                            >
                              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                <rect
                                  x="5.25"
                                  y="5.25"
                                  width="7"
                                  height="7"
                                  rx="1.25"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                />
                                <path
                                  d="M3.75 10.75h-.5c-.55 0-1-.45-1-1v-6.5c0-.55.45-1 1-1h6.5c.55 0 1 .45 1 1v.5"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </AppButton>
                            <AppButton
                              variant="ghost"
                              className="current-box-action is-danger"
                              title="Delete box"
                              aria-label={`Delete ${annotation.label || 'object'}`}
                              onClick={(event) => {
                                event.stopPropagation()
                                removeAnnotation(annotation.id)
                              }}
                            >
                              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                <path
                                  d="M3.5 4.5h9"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M6 2.75h4"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M5 4.5v7.25c0 .41.34.75.75.75h4.5c.41 0 .75-.34.75-.75V4.5"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="M6.75 6.5v4"
                                  stroke="currentColor"
                                  strokeWidth="1.2"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M9.25 6.5v4"
                                  stroke="currentColor"
                                  strokeWidth="1.2"
                                  strokeLinecap="round"
                                />
                              </svg>
                            </AppButton>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span className="muted">No boxes on this image.</span>
                  )}
                </section>
              </>
            ) : (
              <div className="sidebar-plugins-view">
                <div className="sidebar-plugins-toolbar">
                  <p className="section-kicker">Plugins</p>
                </div>

                {plugins.length > 0 ? (
                  <div className="plugin-sidebar-tabs" role="tablist" aria-label="Plugin tabs">
                    {plugins.map((plugin) => (
                      <AppButton
                        key={plugin.id}
                        variant="chip"
                        isActive={selectedPlugin?.id === plugin.id}
                        className="plugin-sidebar-tab"
                        onClick={() => {
                          setSelectedPluginTabId(plugin.id)
                          setSamActionError(null)
                          setSamActionMessage(null)
                        }}
                      >
                        {plugin.name}
                      </AppButton>
                    ))}
                  </div>
                ) : null}

                {selectedPlugin ? (
                  <section className="plugin-card">
                    {selectedPlugin.id === 'sam-3-1' ? (
                      <div className="sam-plugin-panel">
                        <div className="sam-plugin-copy">
                          <div className="plugin-card-title-row">
                            <h3>SAM auto-annotation</h3>
                          </div>
                        </div>

                        <div className="sam-prompt-list" aria-label="SAM prompt and label pairs">
                          {samPromptPairs.map((entry, index) => {
                            const trimmedLabel = entry.label.trim()
                            const usesClassSelect =
                              classList.length > 0 &&
                              (trimmedLabel === '' || classList.includes(trimmedLabel))

                            return (
                              <div
                                key={`sam-prompt-pair-${index}`}
                                className="sam-prompt-row"
                              >
                                <label className="hf-auth-field sam-prompt-field">
                                  <span className="plugin-detail-label">
                                    Prompt {index + 1}
                                  </span>
                                  <input
                                    className="class-manager-input"
                                    type="text"
                                    value={entry.prompt}
                                    onChange={(event) =>
                                      updateSamPromptPair(index, (current) => ({
                                        ...current,
                                        prompt: event.target.value,
                                      }))
                                    }
                                    placeholder="person, red car, traffic light"
                                    autoComplete="off"
                                  />
                                </label>
                                <label className="hf-auth-field sam-prompt-field">
                                  <span className="plugin-detail-label">Label</span>
                                  {usesClassSelect ? (
                                    <select
                                      className="class-manager-input sam-label-select"
                                      value={trimmedLabel}
                                      onChange={(event) =>
                                        updateSamPromptPair(index, (current) => ({
                                          ...current,
                                          label: event.target.value,
                                        }))
                                      }
                                    >
                                      <option value="">Use prompt / active class</option>
                                      {classList.map((label) => (
                                        <option key={label} value={label}>
                                          {label}
                                        </option>
                                      ))}
                                    </select>
                                  ) : (
                                    <input
                                      className="class-manager-input"
                                      type="text"
                                      value={entry.label}
                                      onChange={(event) =>
                                        updateSamPromptPair(index, (current) => ({
                                          ...current,
                                          label: event.target.value,
                                        }))
                                      }
                                      placeholder={effectiveActiveLabel || 'object'}
                                      autoComplete="off"
                                    />
                                  )}
                                </label>
                                <AppButton
                                  variant="ghost"
                                  className="sam-prompt-row-remove"
                                  onClick={() => removeSamPromptPair(index)}
                                  title={
                                    samPromptPairs.length > 1
                                      ? `Remove prompt ${index + 1}`
                                      : 'Clear prompt and label'
                                  }
                                >
                                  {samPromptPairs.length > 1 ? 'Remove' : 'Clear'}
                                </AppButton>
                              </div>
                            )
                          })}
                        </div>

                        <AppButton
                          variant="menu-trigger"
                          className="sam-prompt-add-button"
                          onClick={addSamPromptPair}
                        >
                          Add prompt
                        </AppButton>

                        <div className="hf-auth-config-grid">
                          <label className="hf-auth-field">
                            <span className="plugin-field-header">
                              <span className="plugin-detail-label">
                                Score threshold
                              </span>
                              <strong className="plugin-field-value">
                                {samScoreThresholdValue.toFixed(2)}
                              </strong>
                            </span>
                            <input
                              className="plugin-range-input"
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={samScoreThresholdValue}
                              onChange={(event) =>
                                setSamScoreThresholdDraft(event.target.value)
                              }
                            />
                          </label>
                          <label className="hf-auth-field">
                            <span className="plugin-detail-label">Max results</span>
                            <input
                              className="class-manager-input"
                              type="number"
                              min="1"
                              max="64"
                              step="1"
                              value={samMaxResultsDraft}
                              onChange={(event) =>
                                setSamMaxResultsDraft(event.target.value)
                              }
                            />
                          </label>
                        </div>

                        {selectedAnnotation ? (
                          <p className="plugin-manager-note">
                            Selected box: {selectedAnnotation.label || 'object'} at{' '}
                            {Math.round(selectedAnnotation.x)},{' '}
                            {Math.round(selectedAnnotation.y)} /{' '}
                            {Math.round(selectedAnnotation.width)}x
                            {Math.round(selectedAnnotation.height)}
                          </p>
                        ) : null}

                        {!currentEntry ? (
                          <p className="plugin-manager-note">
                            Open an image before using SAM auto-annotation.
                          </p>
                        ) : null}

                        <div className="hf-auth-actions">
                          <AppButton
                            variant="primary"
                            className="plugin-download-action"
                            onClick={() => void handleRunSamAutoAnnotate('full-image')}
                            disabled={
                              backendStatus !== 'online' ||
                              isRunningSamAction ||
                              !currentEntry ||
                              !selectedPlugin.model.isInstalled ||
                              selectedPlugin.runtime.status !== 'ready'
                            }
                          >
                            {isRunningSamAction ? 'Running SAM...' : 'Auto-annotate image'}
                          </AppButton>
                          <AppButton
                            variant="ghost"
                            className="class-manager-close"
                            onClick={() => void handleRunSamAutoAnnotate('selected-box')}
                            disabled={
                              backendStatus !== 'online' ||
                              isRunningSamAction ||
                              !currentEntry ||
                              !selectedAnnotation ||
                              !selectedPlugin.model.isInstalled ||
                              selectedPlugin.runtime.status !== 'ready'
                            }
                          >
                            Refine selected box
                          </AppButton>
                        </div>
                      </div>
                    ) : null}
                  </section>
                ) : (
                  <span className="muted">No plugins registered.</span>
                )}
              </div>
            )}
          </aside>
        ) : null}
      </div>

      {isClassManagerOpen ? (
        <div
          className="lightbox-backdrop"
          onClick={closeClassManager}
        >
          <div
            className="class-manager-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label="Project classes"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="class-manager-header">
              <div>
                <p className="section-kicker">Settings</p>
                <h2>Project classes</h2>
                <p className="plugin-manager-note">
                  Used for YOLO export and empty-image annotations in the current
                  project.
                </p>
              </div>
              <AppButton
                variant="ghost"
                className="class-manager-close"
                onClick={closeClassManager}
              >
                Close
              </AppButton>
            </div>

            <div className="class-manager-add-row">
              <input
                className="class-manager-input"
                type="text"
                value={newClassName}
                onChange={(event) => setNewClassName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    addClass()
                  }
                }}
                placeholder="New project class"
              />
              <AppButton
                variant="primary"
                className="class-manager-add-action"
                onClick={addClass}
              >
                Add
              </AppButton>
            </div>

            <div className="class-manager-list" aria-label="Class manager list">
              {classList.length > 0 ? (
                classList.map((label) => {
                  const isProjectClass = projectClasses.includes(label)
                  const usageCount = classUsageCounts[label] ?? 0
                  const isEditing = editingClassLabel === label
                  const canDelete = usageCount === 0

                  return (
                    <div key={label} className="class-manager-item">
                      {isEditing ? (
                        <input
                          className="class-manager-input"
                          type="text"
                          value={editingClassDraft}
                          onChange={(event) =>
                            setEditingClassDraft(event.target.value)
                          }
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') {
                              event.preventDefault()
                              renameClass(label)
                            }
                            if (event.key === 'Escape') {
                              event.preventDefault()
                              cancelEditingClass()
                            }
                          }}
                          autoFocus
                        />
                      ) : (
                        <button
                          type="button"
                          className={
                            label === effectiveActiveLabel
                              ? 'class-manager-main is-active'
                              : 'class-manager-main'
                          }
                          onClick={() => setActiveLabel(label)}
                        >
                          <span className="class-manager-name">{label}</span>
                          <span className="class-manager-meta">
                            {isProjectClass ? 'project' : 'annotation only'}
                            {usageCount > 0
                              ? ` · ${usageCount} ${usageCount === 1 ? 'box' : 'boxes'}`
                              : ''}
                          </span>
                        </button>
                      )}

                      <div className="class-manager-actions">
                        {isEditing ? (
                          <>
                            <AppButton
                              variant="ghost"
                              className="class-manager-action"
                              onClick={() => renameClass(label)}
                            >
                              Save
                            </AppButton>
                            <AppButton
                              variant="ghost"
                              className="class-manager-action"
                              onClick={cancelEditingClass}
                            >
                              Cancel
                            </AppButton>
                          </>
                        ) : (
                          <>
                            <AppButton
                              variant="ghost"
                              className="class-manager-action is-icon"
                              onClick={() => startEditingClass(label)}
                              title="Rename class"
                              aria-label={`Rename class ${label}`}
                            >
                              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                <path
                                  d="M3.25 11.5 11.7 3.05a1.5 1.5 0 0 1 2.12 0l.13.13a1.5 1.5 0 0 1 0 2.12L5.5 13.75l-2.75.5.5-2.75Z"
                                  stroke="currentColor"
                                  strokeWidth="1.3"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="m10.75 4 1.25 1.25"
                                  stroke="currentColor"
                                  strokeWidth="1.3"
                                  strokeLinecap="round"
                                />
                              </svg>
                            </AppButton>
                            <AppButton
                              variant="ghost"
                              className="class-manager-action is-danger is-icon"
                              onClick={() => removeClass(label)}
                              disabled={!canDelete}
                              title={
                                usageCount > 0
                                  ? 'Reassign or delete boxes before removing this class'
                                  : 'Delete class'
                              }
                              aria-label={`Delete class ${label}`}
                            >
                              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                <path
                                  d="M3.5 4.5h9"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M6 2.75h4"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M5 4.5v7.25c0 .41.34.75.75.75h4.5c.41 0 .75-.34.75-.75V4.5"
                                  stroke="currentColor"
                                  strokeWidth="1.35"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="M6.75 6.5v4"
                                  stroke="currentColor"
                                  strokeWidth="1.2"
                                  strokeLinecap="round"
                                />
                                <path
                                  d="M9.25 6.5v4"
                                  stroke="currentColor"
                                  strokeWidth="1.2"
                                  strokeLinecap="round"
                                />
                              </svg>
                            </AppButton>
                          </>
                        )}
                      </div>
                    </div>
                  )
                })
              ) : (
                <span className="muted">No project classes configured.</span>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {isPluginsOpen ? (
        <div
          className="lightbox-backdrop"
          onClick={closePluginsManager}
        >
          <div
            className="plugin-manager-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label="Plugins"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="class-manager-header">
              <div>
                <p className="section-kicker">Plugins</p>
              </div>
              <div className="hotkeys-toolbar">
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={() => {
                    void refreshPlugins()
                    void refreshHfAuthStatus()
                  }}
                  disabled={backendStatus !== 'online' || downloadingPluginId !== null}
                >
                  Refresh
                </AppButton>
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={closePluginsManager}
                >
                  Close
                </AppButton>
              </div>
            </div>

            <section className="hf-auth-panel" aria-label="Hugging Face authorization">
              <div className="hf-auth-copy">
                <div className="plugin-card-title-row">
                  <h3>Hugging Face access</h3>
                  <span
                    className={
                      hfAuthStatus?.hasUsableAccessToken
                        ? 'plugin-status is-installed'
                        : 'plugin-status'
                    }
                  >
                    {hfAuthStatus?.authSource === 'oauth'
                      ? 'Connected'
                      : hfAuthStatus?.isExpired
                        ? 'Expired'
                        : hfAuthStatus?.isConfigured
                          ? 'Not connected'
                          : 'Not configured'}
                  </span>
                </div>
                {hfAuthStatus?.authSource === 'oauth' ? null : (
                  <p className="plugin-manager-note">
                    {hfAuthStatus?.isConfigured
                      ? 'Save the OAuth Client ID once, then connect with Hugging Face.'
                      : 'Create a public Hugging Face OAuth app, paste its Client ID here, save it, and then click Connect Hugging Face.'}
                  </p>
                )}
                <div className="hf-auth-config-grid">
                  <label className="hf-auth-field">
                    <span className="plugin-detail-label">Client ID</span>
                    <div className="hf-auth-field-row">
                      <input
                        className="class-manager-input"
                        type={isHfClientIdVisible ? 'text' : 'password'}
                        value={hfClientIdDraft}
                        onChange={(event) => setHfClientIdDraft(event.target.value)}
                        placeholder="Paste Hugging Face OAuth Client ID"
                        autoComplete="off"
                        spellCheck={false}
                      />
                      <AppButton
                        variant="ghost"
                        className="class-manager-close hf-auth-field-button"
                        onClick={() =>
                          setIsHfClientIdVisible((current) => !current)
                        }
                        disabled={!hfClientIdDraft}
                      >
                        {isHfClientIdVisible ? 'Hide' : 'Show'}
                      </AppButton>
                    </div>
                  </label>
                  <label className="hf-auth-field">
                    <span className="plugin-detail-label">Callback URL</span>
                    <div className="hf-auth-field-row">
                      <input
                        className="class-manager-input"
                        type="text"
                        value={hfAuthStatus?.callbackUrl ?? ''}
                        readOnly
                      />
                      <AppButton
                        variant="menu-trigger"
                        className="class-manager-close hf-auth-copy-icon-button"
                        onClick={() => void handleCopyHfCallbackUrl()}
                        disabled={!hfAuthStatus?.callbackUrl}
                        aria-label={
                          isCallbackCopied ? 'Callback URL copied' : 'Copy callback URL'
                        }
                        title={
                          isCallbackCopied ? 'Callback URL copied' : 'Copy callback URL'
                        }
                      >
                        {isCallbackCopied ? (
                          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                            <path
                              d="M3.5 8.5 6.5 11.5 12.5 4.5"
                              stroke="currentColor"
                              strokeWidth="1.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        ) : (
                          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                            <rect
                              x="5"
                              y="3"
                              width="7"
                              height="9"
                              rx="1.25"
                              stroke="currentColor"
                              strokeWidth="1.25"
                            />
                            <path
                              d="M4 5H3.25C2.56 5 2 5.56 2 6.25v6.5C2 13.44 2.56 14 3.25 14h5.5C9.44 14 10 13.44 10 12.75V12"
                              stroke="currentColor"
                              strokeWidth="1.25"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        )}
                        <span className="visually-hidden">
                          {isCallbackCopied ? 'Copied' : 'Copy'}
                        </span>
                      </AppButton>
                    </div>
                  </label>
                </div>
                <div className="hf-auth-steps">
                  <p className="plugin-manager-note">
                    1. Open{' '}
                    <a
                      className="plugin-link"
                      href={HUGGING_FACE_OAUTH_APP_URL}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Hugging Face OAuth app settings
                    </a>{' '}
                    and create a public app.
                  </p>
                  <p className="plugin-manager-note">
                    2. Add the callback URL shown above to Redirect URLs.
                  </p>
                  <p className="plugin-manager-note">
                    3. Copy the Client ID here, save it, then click Connect Hugging Face.
                  </p>
                </div>
                {hfAuthStatus?.authSource === 'oauth' ? (
                  <p className="plugin-manager-note">
                    Expires: {formatPluginInstallDate(hfAuthStatus.expiresAt)}
                  </p>
                ) : null}
              </div>
              <div className="hf-auth-actions">
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={() => void handleSaveHuggingFaceAuthConfig()}
                  disabled={backendStatus !== 'online' || isSavingHfConfig}
                >
                  {isSavingHfConfig ? 'Saving...' : 'Save client ID'}
                </AppButton>
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={() => void handleClearHuggingFaceAuthConfig()}
                  disabled={
                    backendStatus !== 'online' ||
                    isSavingHfConfig ||
                    (!hfClientIdDraft.trim() && !hfAuthStatus?.clientId)
                  }
                >
                  Clear config
                </AppButton>
                <AppButton
                  variant="primary"
                  className="plugin-download-action"
                  onClick={() => void handleStartHuggingFaceAuth()}
                  disabled={
                    backendStatus !== 'online' ||
                    isStartingHfAuth ||
                    isSavingHfConfig ||
                    !hfAuthStatus?.isConfigured
                  }
                >
                  {isStartingHfAuth ? 'Opening login...' : 'Connect Hugging Face'}
                </AppButton>
                {hfAuthStatus?.authSource === 'oauth' ? (
                  <AppButton
                    variant="ghost"
                    className="class-manager-close"
                    onClick={() => void handleLogoutHuggingFaceAuth()}
                    disabled={isStartingHfAuth}
                  >
                    Disconnect
                  </AppButton>
                ) : null}
              </div>
            </section>
            <div className="plugin-card-list" aria-label="Installed plugins">
              {plugins.length > 0 ? (
                plugins.map((plugin) => {
                  const pluginDownload = getPluginDownloadState(plugin)
                  const pluginRuntimeInstall = getPluginRuntimeInstallState(plugin)
                  const isDownloading = downloadingPluginId === plugin.id
                  const isRuntimeInstallPending =
                    installingPluginRuntimeId === plugin.id ||
                    pluginRuntimeInstall.status === 'running'
                  const runtimeInstallProfile =
                    pluginRuntimeInstall.resolvedProfile ??
                    pluginRuntimeInstall.requestedProfile
                  const runtimeStatusLabel = getPluginRuntimeStatusLabel(plugin)
                  const isRuntimeReadyOnGpu =
                    plugin.runtime.status === 'ready' &&
                    plugin.runtime.device === 'cuda'
                  const isRuntimeReadyOnCpu =
                    plugin.runtime.status === 'ready' &&
                    plugin.runtime.device === 'cpu'
                  const isCpuRuntimeSupported = plugin.id !== 'sam-3-1'
                  const isRuntimeInstallLocked =
                    backendStatus !== 'online' ||
                    (isAnyPluginRuntimeInstallRunning && !isRuntimeInstallPending)
                  const installStatus = plugin.model.isInstalled
                    ? 'Installed'
                    : 'Model missing'
                  const downloadTotalBytes =
                    pluginDownload.totalBytes ?? plugin.model.expectedSizeBytes
                  const downloadProgressRatio = getProgressRatio(
                    pluginDownload.downloadedBytes,
                    downloadTotalBytes,
                  )
                  const downloadProgressLabel = pluginDownload.totalBytes
                    ? `${formatByteSize(pluginDownload.downloadedBytes)} / ${formatByteSize(pluginDownload.totalBytes)}`
                    : `${formatByteSize(pluginDownload.downloadedBytes)} downloaded`
                  const downloadSpeedBytesPerSecond =
                    getPluginDownloadSpeedBytesPerSecond(pluginDownload)
                  const downloadEtaSeconds = getPluginDownloadEtaSeconds(
                    pluginDownload,
                    downloadTotalBytes,
                  )
                  const downloadElapsedSeconds = getElapsedSeconds(
                    pluginDownload.startedAt,
                  )
                  const runtimeElapsedSeconds = getElapsedSeconds(
                    pluginRuntimeInstall.startedAt,
                  )
                  const runtimeTerminalOutput =
                    getPluginRuntimeInstallTerminalOutput(pluginRuntimeInstall)
                  const shouldShowRuntimeTerminal =
                    runtimeTerminalOutput !== null ||
                    pluginRuntimeInstall.status === 'running'

                  return (
                    <section key={plugin.id} className="plugin-card">
                      <div className="plugin-card-header">
                        <div className="plugin-card-copy">
                          <div className="plugin-card-title-row">
                            <h3>{plugin.name}</h3>
                            <span
                              className={
                                plugin.model.isInstalled
                                  ? 'plugin-status is-installed'
                                  : 'plugin-status'
                              }
                            >
                              {installStatus}
                            </span>
                          </div>
                        </div>
                        <AppButton
                          variant="primary"
                          className="plugin-download-action"
                          onClick={() => void handleDownloadPlugin(plugin.id)}
                          disabled={
                            backendStatus !== 'online' ||
                            plugin.model.isInstalled ||
                            downloadingPluginId !== null ||
                            (plugin.model.requiresAuth &&
                              !hfAuthStatus?.hasUsableAccessToken)
                          }
                        >
                          {isDownloading
                            ? downloadTotalBytes
                              ? `Downloading ${formatProgressPercent(
                                  pluginDownload.downloadedBytes,
                                  downloadTotalBytes,
                                )}`
                              : 'Downloading...'
                            : plugin.model.isInstalled
                              ? 'Installed'
                              : 'Download model'}
                        </AppButton>
                      </div>

                      {pluginDownload.status === 'running' ? (
                        <div className="plugin-progress-panel" role="status" aria-live="polite">
                          <div className="plugin-progress-header">
                            <span className="plugin-progress-label">
                              <span
                                className="dataset-progress-spinner"
                                aria-hidden="true"
                              />
                              Model download
                            </span>
                            <strong className="plugin-progress-value">
                              {formatProgressPercentFromRatio(downloadProgressRatio) ??
                                'In progress'}
                            </strong>
                          </div>
                          <div
                            className="plugin-progress-track"
                            role="progressbar"
                            aria-label={`${plugin.name} model download progress`}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={
                              downloadProgressRatio === null
                                ? undefined
                                : Math.round(downloadProgressRatio * 100)
                            }
                          >
                            <div
                              className={`plugin-progress-fill${
                                downloadProgressRatio === null
                                  ? ' is-indeterminate'
                                  : ''
                              }`}
                              style={{
                                width:
                                  downloadProgressRatio === null
                                    ? '38%'
                                    : `${Math.max(
                                        downloadProgressRatio * 100,
                                        pluginDownload.downloadedBytes > 0 ? 4 : 0,
                                      )}%`,
                              }}
                            />
                          </div>
                          <div className="plugin-progress-meta">
                            <span>{downloadProgressLabel}</span>
                            {downloadSpeedBytesPerSecond ? (
                              <span>{formatByteSize(downloadSpeedBytesPerSecond)}/s</span>
                            ) : null}
                            {downloadEtaSeconds !== null ? (
                              <span>
                                {downloadEtaSeconds > 0
                                  ? `ETA ${formatDurationCompact(downloadEtaSeconds)}`
                                  : 'Finishing...'}
                              </span>
                            ) : downloadElapsedSeconds !== null ? (
                              <span>
                                Elapsed {formatDurationCompact(downloadElapsedSeconds)}
                              </span>
                            ) : null}
                          </div>
                        </div>
                      ) : null}
                      {plugin.runtime.supportsAutoAnnotate ? (
                        <div className="plugin-runtime-panel">
                          <div className="plugin-card-title-row">
                            <span className="plugin-detail-label">Runtime</span>
                            <span
                              className={
                                plugin.runtime.status === 'ready'
                                  ? 'plugin-status is-installed'
                                  : 'plugin-status'
                              }
                            >
                              {runtimeStatusLabel}
                            </span>
                          </div>

                          <div className="plugin-runtime-actions">
                            <AppButton
                              variant={isRuntimeReadyOnGpu ? 'ghost' : 'primary'}
                              className="plugin-runtime-action"
                              onClick={() =>
                                void handleInstallPluginRuntime(plugin.id, 'cuda')
                              }
                              disabled={
                                isRuntimeInstallLocked ||
                                isRuntimeReadyOnGpu ||
                                isRuntimeInstallPending
                              }
                            >
                              {isRuntimeInstallPending &&
                              runtimeInstallProfile === 'cuda'
                                ? 'Installing GPU/CUDA...'
                                : isRuntimeReadyOnGpu
                                  ? 'Installed on GPU'
                                  : 'Install GPU/CUDA'}
                            </AppButton>
                            <AppButton
                              variant="ghost"
                              className="plugin-runtime-action"
                              onClick={() =>
                                void handleInstallPluginRuntime(plugin.id, 'cpu')
                              }
                              disabled={
                                isRuntimeInstallLocked ||
                                !isCpuRuntimeSupported ||
                                isRuntimeReadyOnCpu ||
                                isRuntimeInstallPending
                              }
                            >
                              {isRuntimeInstallPending &&
                              runtimeInstallProfile === 'cpu'
                                ? 'Installing CPU...'
                                : !isCpuRuntimeSupported
                                  ? 'CPU unsupported'
                                  : isRuntimeReadyOnCpu
                                  ? 'Installed on CPU'
                                  : 'Install CPU'}
                            </AppButton>
                          </div>

                          {pluginRuntimeInstall.status === 'running' ? (
                            shouldShowRuntimeTerminal ? (
                              <div
                                className="plugin-runtime-terminal"
                                role="log"
                                aria-live="polite"
                                aria-atomic="false"
                              >
                                <div className="plugin-runtime-terminal-header">
                                  <span className="plugin-progress-label">
                                    <span
                                      className="dataset-progress-spinner"
                                      aria-hidden="true"
                                    />
                                    Install terminal
                                  </span>
                                  <strong className="plugin-progress-value">
                                    {formatPluginRuntimeInstallStep(
                                      pluginRuntimeInstall.step,
                                    )}
                                  </strong>
                                </div>
                                <div className="plugin-progress-meta">
                                  <span>
                                    Profile:{' '}
                                    {formatPluginRuntimeProfileLabel(
                                      runtimeInstallProfile,
                                    )}
                                  </span>
                                  {runtimeElapsedSeconds !== null ? (
                                    <span>
                                      Elapsed{' '}
                                      {formatDurationCompact(runtimeElapsedSeconds)}
                                    </span>
                                  ) : null}
                                </div>
                                <pre
                                  ref={
                                    isRuntimeInstallPending
                                      ? runtimeInstallLogRef
                                      : undefined
                                  }
                                  className="plugin-runtime-terminal-body"
                                >
                                  {runtimeTerminalOutput ??
                                    'Waiting for installation output...\n'}
                                </pre>
                              </div>
                            ) : null
                          ) : null}
                          {pluginRuntimeInstall.status !== 'running' &&
                          shouldShowRuntimeTerminal ? (
                            <div className="plugin-runtime-terminal">
                              <div className="plugin-runtime-terminal-header">
                                <span className="plugin-progress-label">
                                  Install terminal
                                </span>
                                <strong className="plugin-progress-value">
                                  {pluginRuntimeInstall.status === 'completed'
                                    ? 'Completed'
                                    : pluginRuntimeInstall.status === 'failed'
                                      ? 'Failed'
                                      : 'Idle'}
                                </strong>
                              </div>
                              <pre className="plugin-runtime-terminal-body">
                                {runtimeTerminalOutput}
                              </pre>
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      <div className="plugin-chip-list" aria-label={`${plugin.name} capabilities`}>
                        {plugin.capabilities.map((capability) => (
                          <span key={capability} className="plugin-chip">
                            {capability}
                          </span>
                        ))}
                      </div>

                      <div className="plugin-details-grid">
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Target</span>
                          <strong>{plugin.integrationTarget}</strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Version</span>
                          <strong>{plugin.version}</strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Model file</span>
                          <strong>{plugin.model.filename}</strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Size</span>
                          <strong>
                            {formatByteSize(
                              plugin.model.installedBytes ??
                                plugin.model.expectedSizeBytes,
                            )}
                          </strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Installed at</span>
                          <strong>
                            {formatPluginInstallDate(plugin.model.installedAt)}
                          </strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Auth</span>
                          <strong>
                            {plugin.model.requiresAuth
                              ? hfAuthStatus?.hasUsableAccessToken
                                ? 'Ready via OAuth'
                                : 'Connect HF account'
                              : 'Not required'}
                          </strong>
                        </div>
                        <div className="plugin-detail">
                          <span className="plugin-detail-label">Runtime</span>
                          <strong>{runtimeStatusLabel}</strong>
                        </div>
                      </div>

                    </section>
                  )
                })
              ) : (
                <span className="muted">No plugins registered.</span>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {isHotkeysOpen ? (
        <div
          className="lightbox-backdrop"
          onClick={closeHotkeys}
        >
          <div
            className="hotkeys-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label="Hotkeys"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="class-manager-header">
              <div>
                <p className="section-kicker">Settings</p>
                <h2>Hotkeys</h2>
              </div>
              <div className="hotkeys-toolbar">
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={resetHotkeysToDefaults}
                >
                  Reset defaults
                </AppButton>
                <AppButton
                  variant="ghost"
                  className="class-manager-close"
                  onClick={closeHotkeys}
                >
                  Close
                </AppButton>
              </div>
            </div>

            <div className="hotkeys-sections" aria-label="Keyboard shortcuts">
              {hotkeySections.map((section) => (
                <section key={section.title} className="hotkeys-section">
                  <div className="hotkeys-section-title">{section.title}</div>
                  <div className="hotkeys-list">
                    {section.items.map((item) => {
                      const bindings = hotkeyBindings[item.id]
                      const isCapturing =
                        hotkeyCaptureTarget?.actionId === item.id

                      return (
                        <div
                          key={item.id}
                          className={
                            isCapturing ? 'hotkeys-row is-capturing' : 'hotkeys-row'
                          }
                        >
                          <div className="hotkeys-meta">
                            <div className="hotkeys-action-title">
                              {item.title}
                            </div>
                            <span className="hotkeys-description">
                              {item.description}
                            </span>
                          </div>

                          <div className="hotkeys-binding-stack">
                            {bindings.length > 0 ? (
                              bindings.map((binding, index) => (
                                <div
                                  key={`${item.id}-${binding}-${index}`}
                                  className="hotkeys-binding"
                                >
                                  <kbd className="hotkeys-keys">{binding}</kbd>
                                  <div className="hotkeys-binding-actions">
                                    <AppButton
                                      variant="ghost"
                                      className={
                                        isCapturing &&
                                        hotkeyCaptureTarget?.bindingIndex === index
                                          ? 'hotkeys-binding-action'
                                          : 'hotkeys-binding-action is-icon'
                                      }
                                      onClick={() =>
                                        beginHotkeyCapture(item.id, index)
                                      }
                                      title={
                                        isCapturing &&
                                        hotkeyCaptureTarget?.bindingIndex === index
                                          ? 'Press the new keys'
                                          : 'Edit shortcut'
                                      }
                                      aria-label={
                                        isCapturing &&
                                        hotkeyCaptureTarget?.bindingIndex === index
                                          ? 'Press the new keys'
                                          : `Edit shortcut for ${item.title}`
                                      }
                                    >
                                      {isCapturing &&
                                      hotkeyCaptureTarget?.bindingIndex === index
                                        ? 'Press keys…'
                                        : (
                                            <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                              <path
                                                d="M3.25 11.5 11.7 3.05a1.5 1.5 0 0 1 2.12 0l.13.13a1.5 1.5 0 0 1 0 2.12L5.5 13.75l-2.75.5.5-2.75Z"
                                                stroke="currentColor"
                                                strokeWidth="1.3"
                                                strokeLinejoin="round"
                                              />
                                              <path
                                                d="m10.75 4 1.25 1.25"
                                                stroke="currentColor"
                                                strokeWidth="1.3"
                                                strokeLinecap="round"
                                              />
                                            </svg>
                                          )}
                                    </AppButton>
                                    <AppButton
                                      variant="ghost"
                                      className="hotkeys-binding-action is-danger is-icon"
                                      onClick={() =>
                                        removeHotkeyBinding(item.id, index)
                                      }
                                      title="Remove shortcut"
                                      aria-label={`Remove shortcut ${binding} for ${item.title}`}
                                    >
                                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                        <path
                                          d="M3.5 4.5h9"
                                          stroke="currentColor"
                                          strokeWidth="1.35"
                                          strokeLinecap="round"
                                        />
                                        <path
                                          d="M6 2.75h4"
                                          stroke="currentColor"
                                          strokeWidth="1.35"
                                          strokeLinecap="round"
                                        />
                                        <path
                                          d="M5 4.5v7.25c0 .41.34.75.75.75h4.5c.41 0 .75-.34.75-.75V4.5"
                                          stroke="currentColor"
                                          strokeWidth="1.35"
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                        />
                                        <path
                                          d="M6.75 6.5v4"
                                          stroke="currentColor"
                                          strokeWidth="1.2"
                                          strokeLinecap="round"
                                        />
                                        <path
                                          d="M9.25 6.5v4"
                                          stroke="currentColor"
                                          strokeWidth="1.2"
                                          strokeLinecap="round"
                                        />
                                      </svg>
                                    </AppButton>
                                  </div>
                                </div>
                              ))
                            ) : (
                              <span className="muted">
                                No shortcuts assigned.
                              </span>
                            )}

                            <div className="hotkeys-row-actions">
                              <AppButton
                                variant="ghost"
                                className={
                                  isCapturing &&
                                  hotkeyCaptureTarget?.bindingIndex === null
                                    ? 'hotkeys-binding-action'
                                    : 'hotkeys-binding-action is-icon'
                                }
                                onClick={() => beginHotkeyCapture(item.id, null)}
                                title={
                                  isCapturing &&
                                  hotkeyCaptureTarget?.bindingIndex === null
                                    ? 'Press the new keys'
                                    : 'Add shortcut'
                                }
                                aria-label={
                                  isCapturing &&
                                  hotkeyCaptureTarget?.bindingIndex === null
                                    ? 'Press the new keys'
                                    : `Add shortcut for ${item.title}`
                                }
                              >
                                {isCapturing &&
                                hotkeyCaptureTarget?.bindingIndex === null
                                  ? 'Press keys…'
                                  : (
                                      <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                                        <path
                                          d="M8 3v10M3 8h10"
                                          stroke="currentColor"
                                          strokeWidth="1.35"
                                          strokeLinecap="round"
                                        />
                                      </svg>
                                    )}
                              </AppButton>
                              {isCapturing ? (
                                <>
                                  <span className="hotkeys-capture-note">
                                    Press the new key combination now.
                                  </span>
                                  <AppButton
                                    variant="ghost"
                                    className="hotkeys-binding-action"
                                    onClick={cancelHotkeyCapture}
                                  >
                                    Cancel
                                  </AppButton>
                                </>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </section>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {confirmDialogState ? (
        <ConfirmDialog
          title={confirmDialogState.title}
          message={confirmDialogState.message}
          confirmLabel={confirmDialogState.confirmLabel}
          confirmTone={confirmDialogState.confirmTone}
          onCancel={closeConfirmDialog}
          onConfirm={confirmDialogState.onConfirm}
        />
      ) : null}
    </div>
  )
}

export default App

async function waitForSessionJob(
  jobId: string,
  onProgress: (progress: {
    phase: LocalSessionJobResponse['phase']
    processed: number
    total: number
  }) => void,
  onSessionUpdate: (session: LocalSessionResponse) => void,
  onCompleted?: (session: LocalSessionResponse) => void,
) {
  let lastSessionRevision = 0
  let lastSession: LocalSessionResponse | null = null

  while (true) {
    const job = await fetchLocalSessionJob(jobId, lastSessionRevision)
    onProgress({
      phase: job.phase,
      processed: job.processedImages,
      total: job.totalImages,
    })

    if (job.session && !job.session.cancelled) {
      lastSessionRevision = job.sessionRevision
      lastSession = job.session
      onSessionUpdate(job.session)
    }

    if (job.status === 'completed') {
      const completedSession =
        job.session && !job.session.cancelled ? job.session : lastSession

      if (!completedSession || completedSession.cancelled) {
        throw new Error('Directory loading finished without a session')
      }

      onCompleted?.(completedSession)
      return
    }

    if (job.status === 'failed') {
      throw new Error(job.error || 'Failed to open directory')
    }

    await delay(180)
  }
}

async function preloadImage(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () =>
      resolve({
        width: image.naturalWidth,
        height: image.naturalHeight,
      })
    image.onerror = () => reject(new Error('Failed to load image'))
    image.src = url
  })
}

function getBufferedEntries(
  images: ImageEntry[],
  currentIndex: number,
  radius: number,
) {
  const buffered: ImageEntry[] = []
  const currentEntry = images[currentIndex]
  if (currentEntry) {
    buffered.push(currentEntry)
  }

  for (let offset = 1; offset <= radius; offset += 1) {
    const previousEntry = images[currentIndex - offset]
    if (previousEntry) {
      buffered.push(previousEntry)
    }

    const nextEntry = images[currentIndex + offset]
    if (nextEntry) {
      buffered.push(nextEntry)
    }
  }

  return buffered
}

function stripExtension(filename: string) {
  const dotIndex = filename.lastIndexOf('.')
  return dotIndex > 0 ? filename.slice(0, dotIndex) : filename
}

function clampNumericInput(
  value: string,
  fallback: number,
  bounds: { min: number; max: number },
) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return fallback
  }

  return Math.min(bounds.max, Math.max(bounds.min, parsed))
}

function getPluginDownloadState(plugin: PluginInfo) {
  return (
    plugin.download ?? {
      status: 'idle' as const,
      downloadedBytes: 0,
      totalBytes: plugin.model.expectedSizeBytes ?? null,
      error: null,
      startedAt: null,
      finishedAt: null,
    }
  )
}

function getPluginRuntimeInstallState(plugin: PluginInfo) {
  return (
    plugin.runtime.install ?? {
      status: 'idle' as const,
      requestedProfile: null,
      resolvedProfile: null,
      step: null,
      stepStartedAt: null,
      log: null,
      message: null,
      error: null,
      startedAt: null,
      finishedAt: null,
    }
  )
}

function getPluginRuntimeStatusLabel(plugin: PluginInfo) {
  if (plugin.runtime.status === 'ready') {
    if (plugin.runtime.device === 'cuda') {
      return 'Ready on GPU'
    }
    if (plugin.runtime.device === 'cpu') {
      return 'Ready on CPU'
    }
    return 'Ready'
  }

  if (plugin.runtime.status === 'missing-model') {
    return 'Model missing'
  }

  if (plugin.runtime.status === 'missing-runtime') {
    return 'Runtime missing'
  }

  return 'Setup failed'
}

function formatPluginRuntimeProfileLabel(
  profile?: PluginRuntimeInstallProfile | null,
) {
  if (profile === 'cuda') {
    return 'GPU/CUDA'
  }
  if (profile === 'cpu') {
    return 'CPU'
  }
  return 'Auto'
}

function formatPluginRuntimeInstallStep(step?: string | null) {
  switch (step) {
    case 'preparing':
      return 'Preparing environment'
    case 'installing-pytorch':
      return 'Installing PyTorch'
    case 'downloading-sam3':
      return 'Downloading SAM 3 source'
    case 'installing-sam3':
      return 'Installing SAM 3 package'
    case 'verifying':
      return 'Verifying runtime'
    case 'completed':
      return 'Completed'
    case 'failed':
      return 'Failed'
    default:
      return step || 'Preparing environment'
  }
}

function getPluginRuntimeInstallTerminalOutput(
  install: PluginRuntimeInstallState,
) {
  if (install.log && install.log.trim()) {
    return install.log
  }

  if (install.message) {
    return `[status] ${install.message}\n`
  }

  return null
}

function parseIsoTimestamp(value?: string | null) {
  if (!value) {
    return null
  }

  const parsed = new Date(value)
  const time = parsed.getTime()
  if (Number.isNaN(time)) {
    return null
  }

  return time
}

function getElapsedSeconds(startedAt?: string | null) {
  const startedAtMs = parseIsoTimestamp(startedAt)
  if (startedAtMs === null) {
    return null
  }

  return Math.max(0, Math.floor((Date.now() - startedAtMs) / 1000))
}

function formatDurationCompact(value?: number | null) {
  if (value === null || value === undefined || value < 0) {
    return null
  }

  const totalSeconds = Math.max(0, Math.round(value))
  if (totalSeconds < 60) {
    return `${totalSeconds}s`
  }

  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (totalSeconds < 3600) {
    return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`
  }

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`
}

function getProgressRatio(completed: number, total?: number | null) {
  if (!total || total <= 0) {
    return null
  }

  return Math.max(0, Math.min(1, completed / total))
}

function formatByteSize(value?: number | null) {
  if (!value || value <= 0) {
    return '—'
  }

  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = value
  let unitIndex = 0

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }

  const digits = size >= 100 || unitIndex === 0 ? 0 : 1
  return `${size.toFixed(digits)} ${units[unitIndex]}`
}

function formatProgressPercent(downloadedBytes: number, totalBytes: number) {
  if (!totalBytes || totalBytes <= 0) {
    return '0%'
  }

  const percent = Math.max(
    0,
    Math.min(100, Math.round((downloadedBytes / totalBytes) * 100)),
  )
  return `${percent}%`
}

function formatProgressPercentFromRatio(ratio?: number | null) {
  if (ratio === null || ratio === undefined) {
    return null
  }

  return `${Math.max(0, Math.min(100, Math.round(ratio * 100)))}%`
}

function getPluginDownloadSpeedBytesPerSecond(download: PluginDownloadState) {
  const elapsedSeconds = getElapsedSeconds(download.startedAt)
  if (!elapsedSeconds || elapsedSeconds < 1 || download.downloadedBytes <= 0) {
    return null
  }

  return download.downloadedBytes / elapsedSeconds
}

function getPluginDownloadEtaSeconds(
  download: PluginDownloadState,
  totalBytes?: number | null,
) {
  if (!totalBytes || totalBytes <= download.downloadedBytes) {
    return null
  }

  const elapsedSeconds = getElapsedSeconds(download.startedAt)
  if (!elapsedSeconds || elapsedSeconds < 3) {
    return null
  }

  const speedBytesPerSecond = getPluginDownloadSpeedBytesPerSecond(download)
  if (!speedBytesPerSecond || speedBytesPerSecond <= 0) {
    return null
  }

  return Math.max(
    0,
    Math.round((totalBytes - download.downloadedBytes) / speedBytesPerSecond),
  )
}

function formatPluginInstallDate(value?: string | null) {
  if (!value) {
    return 'Not installed'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }

  return parsed.toLocaleString()
}

function createEmptySamPromptPair(): SamPromptPairDraft {
  return {
    prompt: '',
    label: '',
  }
}

function buildDefaultSamSettings(): SamSettingsDraft {
  return {
    entries: [createEmptySamPromptPair()],
    scoreThreshold: DEFAULT_SAM_SCORE_THRESHOLD,
    maxResults: DEFAULT_SAM_MAX_RESULTS,
  }
}

function normalizeSamPromptPairs(value: unknown): SamPromptPairDraft[] {
  if (!Array.isArray(value)) {
    return [createEmptySamPromptPair()]
  }

  const normalized = value
    .filter((entry): entry is Record<string, unknown> => {
      return typeof entry === 'object' && entry !== null
    })
    .map((entry) => ({
      prompt: typeof entry.prompt === 'string' ? entry.prompt : '',
      label: typeof entry.label === 'string' ? entry.label : '',
    }))

  return normalized.length > 0 ? normalized : [createEmptySamPromptPair()]
}

function coerceSamSettings(value: unknown): SamSettingsDraft {
  const defaults = buildDefaultSamSettings()

  if (typeof value !== 'object' || value === null) {
    return defaults
  }

  const record = value as Record<string, unknown>
  return {
    entries: normalizeSamPromptPairs(record.entries),
    scoreThreshold:
      typeof record.scoreThreshold === 'string' &&
      record.scoreThreshold.trim() !== ''
        ? record.scoreThreshold
        : defaults.scoreThreshold,
    maxResults:
      typeof record.maxResults === 'string' && record.maxResults.trim() !== ''
        ? record.maxResults
        : defaults.maxResults,
  }
}

function sanitizeSamSettings(value: SamSettingsDraft): SamSettingsDraft {
  return {
    entries: normalizeSamPromptPairs(value.entries),
    scoreThreshold:
      typeof value.scoreThreshold === 'string' &&
      value.scoreThreshold.trim() !== ''
        ? value.scoreThreshold
        : DEFAULT_SAM_SCORE_THRESHOLD,
    maxResults:
      typeof value.maxResults === 'string' && value.maxResults.trim() !== ''
        ? value.maxResults
        : DEFAULT_SAM_MAX_RESULTS,
  }
}

function readStoredSidebarVisibility() {
  if (typeof window === 'undefined') {
    return true
  }

  try {
    return coerceSidebarVisibility(
      window.localStorage.getItem(SIDEBAR_VISIBILITY_STORAGE_KEY),
    ) ?? true
  } catch {
    return true
  }
}

function readStoredRecentDatasets(): RecentDataset[] {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(RECENT_DATASETS_STORAGE_KEY)
    if (!raw) {
      return []
    }

    return coerceRecentDatasets(JSON.parse(raw) as unknown)
  } catch {
    return []
  }
}

function readStoredPersistedSessionState(): PersistedSessionState | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    const raw = window.localStorage.getItem(SESSION_STATE_STORAGE_KEY)
    if (!raw) {
      return null
    }

    return coercePersistedSessionState(JSON.parse(raw) as unknown)
  } catch {
    return null
  }
}

function readStoredHotkeyBindings(): HotkeyBindings {
  if (typeof window === 'undefined') {
    return buildDefaultHotkeyBindings()
  }

  try {
    const raw = window.localStorage.getItem(HOTKEY_BINDINGS_STORAGE_KEY)
    if (!raw) {
      return buildDefaultHotkeyBindings()
    }

    return coerceHotkeyBindings(JSON.parse(raw) as unknown)
  } catch {
    return buildDefaultHotkeyBindings()
  }
}

function readStoredSamSettings(): SamSettingsDraft {
  if (typeof window === 'undefined') {
    return buildDefaultSamSettings()
  }

  try {
    const raw = window.localStorage.getItem(SAM_SETTINGS_STORAGE_KEY)
    if (!raw) {
      return buildDefaultSamSettings()
    }

    return coerceSamSettings(JSON.parse(raw) as unknown)
  } catch {
    return buildDefaultSamSettings()
  }
}

function coerceSidebarVisibility(value: unknown) {
  if (typeof value === 'boolean') {
    return value
  }

  if (value === '1') {
    return true
  }

  if (value === '0') {
    return false
  }

  return null
}

function coerceRecentDatasets(value: unknown): RecentDataset[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .filter((entry): entry is RecentDataset => {
      return (
        typeof entry === 'object' &&
        entry !== null &&
        'path' in entry &&
        'label' in entry &&
        typeof entry.path === 'string' &&
        typeof entry.label === 'string'
      )
    })
    .slice(0, MAX_RECENT_DATASETS)
}

function coercePersistedSessionState(
  value: unknown,
): PersistedSessionState | null {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('sourceKind' in value) ||
    !('sourcePath' in value)
  ) {
    return null
  }

  const { sourceKind, sourcePath, currentImageRelativePath } = value as {
    sourceKind?: unknown
    sourcePath?: unknown
    currentImageRelativePath?: unknown
  }

  if (
    (sourceKind !== 'image' && sourceKind !== 'dataset') ||
    typeof sourcePath !== 'string' ||
    sourcePath.trim() === ''
  ) {
    return null
  }

  return {
    sourceKind,
    sourcePath,
    currentImageRelativePath:
      typeof currentImageRelativePath === 'string'
        ? currentImageRelativePath
        : null,
  }
}

function coerceProjectClassesByRootPath(value: unknown) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return {}
  }

  const next: Record<string, string[]> = {}

  for (const [rootPath, labels] of Object.entries(value)) {
    if (typeof rootPath !== 'string' || rootPath.trim() === '') {
      continue
    }

    if (!Array.isArray(labels)) {
      continue
    }

    next[rootPath] = sanitizeClassList(
      labels.filter((label): label is string => typeof label === 'string'),
    )
  }

  return next
}

function sanitizeClassList(labels: string[]) {
  return [
    ...new Set(
      labels
        .map((label) => label.trim())
        .filter(Boolean),
    ),
  ]
}

function buildDefaultHotkeyBindings(): HotkeyBindings {
  return HOTKEY_ACTIONS.reduce((current, action) => {
    current[action.id] = [...action.defaultBindings]
    return current
  }, {} as HotkeyBindings)
}

function coerceHotkeyBindings(value: unknown): HotkeyBindings {
  const normalized = buildDefaultHotkeyBindings()
  if (typeof value !== 'object' || value === null) {
    return normalized
  }

  const record = value as Record<string, unknown>
  for (const action of HOTKEY_ACTIONS) {
    const rawBindings = record[action.id]
    if (!Array.isArray(rawBindings)) {
      continue
    }

    normalized[action.id] = dedupeBindings(
      rawBindings
        .filter((binding): binding is string => typeof binding === 'string')
        .map((binding) => normalizeHotkeyString(binding))
        .filter((binding): binding is string => Boolean(binding)),
    )
  }

  return normalized
}

function applyHotkeyCapture(
  current: HotkeyBindings,
  target: HotkeyCaptureTarget,
  binding: string,
): HotkeyBindings {
  const next = HOTKEY_ACTIONS.reduce((accumulator, action) => {
    accumulator[action.id] = current[action.id].filter((existing, index) => {
      if (
        action.id === target.actionId &&
        target.bindingIndex !== null &&
        index === target.bindingIndex
      ) {
        return false
      }

      return existing !== binding
    })
    return accumulator
  }, {} as HotkeyBindings)

  const targetBindings = [...next[target.actionId]]
  const nextIndex =
    target.bindingIndex !== null &&
    target.bindingIndex >= 0 &&
    target.bindingIndex <= targetBindings.length
      ? target.bindingIndex
      : targetBindings.length
  targetBindings.splice(nextIndex, 0, binding)
  next[target.actionId] = dedupeBindings(targetBindings)
  return next
}

function dedupeBindings(bindings: string[]) {
  return [...new Set(bindings)]
}

function normalizeHotkeyEvent(event: KeyboardEvent) {
  if (
    event.code.startsWith('Control') ||
    event.code.startsWith('Shift') ||
    event.code.startsWith('Alt') ||
    event.code.startsWith('Meta')
  ) {
    return null
  }

  const mainKey =
    hotkeyMainKeyFromCode(event.code) ?? normalizeHotkeyMainKey(event.key)
  if (!mainKey) {
    return null
  }

  const modifiers: string[] = []
  if (event.ctrlKey) {
    modifiers.push('Ctrl')
  }
  if (event.altKey) {
    modifiers.push('Alt')
  }
  if (event.shiftKey) {
    modifiers.push('Shift')
  }
  if (event.metaKey) {
    modifiers.push('Meta')
  }

  return [...modifiers, mainKey].join('+')
}

function normalizeHotkeyString(value: string) {
  const parts = value
    .split('+')
    .map((part) => part.trim())
    .filter(Boolean)
  if (parts.length === 0) {
    return null
  }

  const modifiers = new Set<string>()
  let mainKey: string | null = null

  for (const part of parts) {
    const normalizedModifier = normalizeHotkeyModifier(part)
    if (normalizedModifier) {
      modifiers.add(normalizedModifier)
      continue
    }

    const normalizedMainKey = normalizeHotkeyMainKey(part)
    if (!normalizedMainKey || mainKey) {
      return null
    }

    mainKey = normalizedMainKey
  }

  if (!mainKey) {
    return null
  }

  return [
    ...(modifiers.has('Ctrl') ? ['Ctrl'] : []),
    ...(modifiers.has('Alt') ? ['Alt'] : []),
    ...(modifiers.has('Shift') ? ['Shift'] : []),
    ...(modifiers.has('Meta') ? ['Meta'] : []),
    mainKey,
  ].join('+')
}

function normalizeHotkeyModifier(value: string) {
  switch (value.trim().toLowerCase()) {
    case 'ctrl':
    case 'control':
      return 'Ctrl'
    case 'alt':
    case 'option':
      return 'Alt'
    case 'shift':
      return 'Shift'
    case 'cmd':
    case 'command':
    case 'meta':
    case 'win':
    case 'super':
      return 'Meta'
    default:
      return null
  }
}

function hotkeyMainKeyFromCode(code: string) {
  if (code.startsWith('Key')) {
    return code.slice(3).toUpperCase()
  }

  if (code.startsWith('Digit')) {
    return code.slice(5)
  }

  const numpadMatch = code.match(/^Numpad([0-9])$/)
  if (numpadMatch) {
    return numpadMatch[1] ?? null
  }

  switch (code) {
    case 'ArrowLeft':
    case 'ArrowRight':
    case 'ArrowUp':
    case 'ArrowDown':
    case 'Delete':
    case 'Backspace':
    case 'Enter':
    case 'Tab':
      return code
    case 'Escape':
      return 'Esc'
    case 'Space':
      return 'Space'
    case 'Minus':
      return '-'
    case 'Equal':
      return '='
    case 'Comma':
      return ','
    case 'Period':
      return '.'
    case 'Slash':
      return '/'
    case 'Backslash':
      return '\\'
    case 'BracketLeft':
      return '['
    case 'BracketRight':
      return ']'
    case 'Semicolon':
      return ';'
    case 'Quote':
      return "'"
    case 'Backquote':
      return '`'
    default:
      return null
  }
}

function normalizeHotkeyMainKey(value: string) {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  const upper = trimmed.toUpperCase()
  if (/^[A-Z0-9]$/.test(upper)) {
    return upper
  }

  if (/^F(?:[1-9]|1[0-2])$/.test(upper)) {
    return upper
  }

  switch (upper) {
    case 'ESC':
    case 'ESCAPE':
      return 'Esc'
    case 'LEFT':
    case 'ARROWLEFT':
      return 'ArrowLeft'
    case 'RIGHT':
    case 'ARROWRIGHT':
      return 'ArrowRight'
    case 'UP':
    case 'ARROWUP':
      return 'ArrowUp'
    case 'DOWN':
    case 'ARROWDOWN':
      return 'ArrowDown'
    case 'DEL':
    case 'DELETE':
      return 'Delete'
    case 'BACKSPACE':
      return 'Backspace'
    case 'ENTER':
    case 'RETURN':
      return 'Enter'
    case 'TAB':
      return 'Tab'
    case 'SPACE':
    case 'SPACEBAR':
      return 'Space'
    case '-':
    case '=':
    case ',':
    case '.':
    case '/':
    case '\\':
    case '[':
    case ']':
    case ';':
    case "'":
    case '`':
      return trimmed
    default:
      return null
  }
}

function buildPersistedSessionState(
  session: LocalSessionResponse,
  sourceKind: PersistedSessionSourceKind,
): PersistedSessionState | null {
  if (!session.rootPath) {
    return null
  }

  if (sourceKind === 'dataset') {
    return {
      sourceKind,
      sourcePath: session.rootPath,
      currentImageRelativePath: null,
    }
  }

  const firstImage = session.images?.[0]
  if (!firstImage) {
    return null
  }

  return {
    sourceKind,
    sourcePath: joinLocalPath(session.rootPath, firstImage.relativePath),
    currentImageRelativePath: firstImage.relativePath,
  }
}

function joinLocalPath(basePath: string, relativePath: string) {
  const normalizedBase = basePath.replace(/[\\/]+$/, '')
  const normalizedRelative = relativePath.replace(/^[/\\]+/, '')
  if (!normalizedBase) {
    return normalizedRelative
  }

  if (!normalizedRelative) {
    return normalizedBase
  }

  const separator = normalizedBase.includes('\\') ? '\\' : '/'
  const joinedRelative =
    separator === '\\'
      ? normalizedRelative.replace(/\//g, '\\')
      : normalizedRelative.replace(/\\/g, '/')

  return `${normalizedBase}${separator}${joinedRelative}`
}

function arePersistedSessionStatesEqual(
  left: PersistedSessionState | null,
  right: PersistedSessionState | null,
) {
  if (left === right) {
    return true
  }

  if (!left || !right) {
    return false
  }

  return (
    left.sourceKind === right.sourceKind &&
    left.sourcePath === right.sourcePath &&
    left.currentImageRelativePath === right.currentImageRelativePath
  )
}

function labelFromDatasetPath(path: string) {
  const normalized = path.replace(/[\\/]+$/, '')
  const parts = normalized.split(/[\\/]/).filter(Boolean)
  return parts.at(-1) || path
}

function resolveProjectClassAlias(label: string, projectClasses: string[]) {
  const normalizedLabel = label.trim()
  const match = normalizedLabel.match(/^class_(\d+)$/)
  if (!match) {
    return normalizedLabel || 'object'
  }

  const classIndex = Number(match[1])
  if (!Number.isInteger(classIndex) || classIndex < 0) {
    return normalizedLabel || 'object'
  }

  return projectClasses[classIndex] ?? (normalizedLabel || 'object')
}

function remapProjectClassAliasesInImageMap(
  annotationsByImage: Record<string, Annotation[]>,
  projectClasses: string[],
) {
  let hasChanges = false
  const nextEntries = Object.entries(annotationsByImage).map(
    ([imageId, imageAnnotations]) => {
      const nextAnnotations = imageAnnotations.map((annotation) => {
        const label = resolveProjectClassAlias(annotation.label, projectClasses)
        if (label === annotation.label) {
          return annotation
        }

        hasChanges = true
        return {
          ...annotation,
          label,
          color: labelToColor(label),
        }
      })

      return [imageId, nextAnnotations] as const
    },
  )

  return hasChanges ? Object.fromEntries(nextEntries) : annotationsByImage
}

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  const tagName = target.tagName
  return (
    target.isContentEditable ||
    tagName === 'INPUT' ||
    tagName === 'TEXTAREA' ||
    tagName === 'SELECT'
  )
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}
