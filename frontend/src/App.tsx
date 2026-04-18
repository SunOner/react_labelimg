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
  fetchAppState,
  fetchApiHealth,
  fetchLocalAnnotations,
  fetchLocalSessionJob,
  openLocalDirectoryPathJob,
  fetchPredefinedClasses,
  openLocalDirectory,
  openLocalImage,
  openLocalImagePath,
  updateAppState,
  type LocalSessionJobResponse,
  type LocalSessionResponse,
} from './lib/api'
import {
  MIN_BOX_SIZE,
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

const PRELOAD_RADIUS = 1
const SIDEBAR_VISIBILITY_STORAGE_KEY = 'labelimg.sidebarVisible'
const RECENT_DATASETS_STORAGE_KEY = 'labelimg.recentDatasets'
const SESSION_STATE_STORAGE_KEY = 'labelimg.sessionState'
const HOTKEY_BINDINGS_STORAGE_KEY = 'labelimg.hotkeys'
const MAX_RECENT_DATASETS = 6
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
  const [recentDatasets, setRecentDatasets] = useState(readStoredRecentDatasets)
  const [persistedSessionState, setPersistedSessionState] = useState<
    PersistedSessionState | null
  >(readStoredPersistedSessionState)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draftRect, setDraftRect] = useState<Rect | null>(null)
  const [drawStart, setDrawStart] = useState<Point | null>(null)
  const [activeLabel, setActiveLabel] = useState('object')
  const [openMenu, setOpenMenu] = useState<
    'file' | 'annotation' | 'export' | 'settings' | null
  >(null)
  const [backendStatus, setBackendStatus] = useState<
    'checking' | 'online' | 'offline'
  >('checking')
  const [backendClasses, setBackendClasses] = useState<string[]>([])
  const [customClasses, setCustomClasses] = useState<string[]>([])
  const [hotkeyBindings, setHotkeyBindings] = useState<HotkeyBindings>(
    readStoredHotkeyBindings,
  )
  const [hotkeyCaptureTarget, setHotkeyCaptureTarget] =
    useState<HotkeyCaptureTarget | null>(null)
  const [confirmDialogState, setConfirmDialogState] =
    useState<ConfirmDialogState | null>(null)
  const [isClassManagerOpen, setIsClassManagerOpen] = useState(false)
  const [isHotkeysOpen, setIsHotkeysOpen] = useState(false)
  const [newClassName, setNewClassName] = useState('')
  const [editingClassLabel, setEditingClassLabel] = useState<string | null>(null)
  const [editingClassDraft, setEditingClassDraft] = useState('')
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
  const [draggedAnnotationId, setDraggedAnnotationId] = useState<string | null>(
    null,
  )
  const [dragInsertIndex, setDragInsertIndex] = useState<number | null>(null)

  const menuBarRef = useRef<HTMLElement | null>(null)
  const imageResourcesRef = useRef<Record<string, ImageResource>>({})
  const imageIdSetRef = useRef<Set<string>>(new Set())
  const pendingLoadsRef = useRef<Record<string, Promise<void>>>({})
  const pendingAnnotationLoadsRef = useRef<Record<string, Promise<void>>>({})
  const annotationLoadStateRef = useRef<
    Record<string, 'idle' | 'loading' | 'ready' | 'error'>
  >({})
  const appStateSyncReadyRef = useRef(false)
  const pendingPreferredImageRelativePathRef = useRef<string | null>(null)
  const sessionVersionRef = useRef(0)
  const restoreAttemptedRef = useRef(false)
  const isMountedRef = useRef(true)

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
  const annotationClassList = buildClassList(sessionAnnotations)
  const classList = [
    ...new Set([...backendClasses, ...customClasses, ...annotationClassList]),
  ]
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

  const commitPersistedSessionState = (next: PersistedSessionState | null) => {
    setPersistedSessionState((current) => {
      if (arePersistedSessionStatesEqual(current, next)) {
        return current
      }

      return next
    })
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

  const commitImageResources = (
    updater: (current: Record<string, ImageResource>) => Record<string, ImageResource>,
  ) => {
    setImageResources((current) => {
      const next = updater(current)
      imageResourcesRef.current = next
      return next
    })
  }

  const resetSessionState = (nextImages: ImageEntry[], nextSessionLabel: string) => {
    sessionVersionRef.current += 1
    imageIdSetRef.current = new Set(nextImages.map((entry) => entry.id))
    pendingLoadsRef.current = {}
    pendingAnnotationLoadsRef.current = {}
    annotationLoadStateRef.current = {}
    imageResourcesRef.current = {}
    pendingPreferredImageRelativePathRef.current = null

    startTransition(() => {
      setImageResources({})
      setImages(nextImages)
      setAnnotationsByImage({})
      setCurrentSessionId(null)
      setCurrentImageEntry(nextImages[0] ?? null)
      setSessionLabel(nextSessionLabel)
      setSessionError(null)
      setSelectedId(null)
      setDrawStart(null)
      setDraftRect(null)
      setCustomClasses([])
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
    }
  }, [])

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

        const nextAnnotations = payload.annotations.map((annotation) => ({
          ...annotation,
          color: labelToColor(annotation.label),
        }))

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

        setAnnotationsByImage((current) => {
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
        const [classes, appState] = await Promise.all([
          fetchPredefinedClasses(controller.signal),
          fetchAppState(controller.signal).catch(() => null),
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
        }

        setBackendClasses(classes)
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
    setDrawStart(null)
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
    setDrawStart(null)
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
    setDrawStart(null)
    setDraftRect(null)
    setCurrentImageEntry(nextEntry)
  }

  const clearSession = () => {
    setOpenMenu(null)
    commitPersistedSessionState(null)
    resetSessionState([], 'No session')
  }

  const openClassManager = () => {
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
      setCustomClasses((current) => [...current, nextLabel])
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

    if (backendClasses.includes(sourceLabel)) {
      return
    }

    const hasConflict = classList.some(
      (label) => label === nextLabel && label !== sourceLabel,
    )
    if (hasConflict) {
      return
    }

    const sourceAnnotationCount = classUsageCounts[sourceLabel] ?? 0

    setCustomClasses((current) => {
      const next = current.filter(
        (label) => label !== sourceLabel && label !== nextLabel,
      )
      if (current.includes(sourceLabel) || sourceAnnotationCount === 0) {
        next.push(nextLabel)
      }
      return next
    })

    setAnnotationsByImage((current) =>
      Object.fromEntries(
        Object.entries(current).map(([imageId, imageAnnotations]) => [
          imageId,
          imageAnnotations.map((annotation) =>
            annotation.label === sourceLabel
              ? {
                  ...annotation,
                  label: nextLabel,
                  color: labelToColor(nextLabel),
                }
              : annotation,
          ),
        ]),
      ),
    )

    setActiveLabel((current) =>
      current === sourceLabel ? nextLabel : current,
    )
    cancelEditingClass()
  }

  const removeClass = (label: string) => {
    if (backendClasses.includes(label) || (classUsageCounts[label] ?? 0) > 0) {
      return
    }

    setCustomClasses((current) => current.filter((entry) => entry !== label))
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

    annotationLoadStateRef.current[currentEntry.id] = 'ready'

    setAnnotationsByImage((current) => ({
      ...current,
      [currentEntry.id]: (current[currentEntry.id] ?? []).filter(
        (annotation) => annotation.id !== annotationId,
      ),
    }))
    setSelectedId((current) => (current === annotationId ? null : current))
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

    annotationLoadStateRef.current[currentEntry.id] = 'ready'

    setAnnotationsByImage((current) => ({
      ...current,
      [currentEntry.id]: [...(current[currentEntry.id] ?? []), nextAnnotation],
    }))
    setSelectedId(nextAnnotation.id)
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

    annotationLoadStateRef.current[currentEntry.id] = 'ready'

    setAnnotationsByImage((current) => ({
      ...current,
      [currentEntry.id]: (current[currentEntry.id] ?? []).map((annotation) =>
        annotation.id === annotationId
          ? {
              ...annotation,
              ...nextRect,
            }
          : annotation,
      ),
    }))
  })

  const moveCurrentImageAnnotation = (
    annotationId: string,
    insertIndex: number,
  ) => {
    if (!currentEntry) {
      return
    }

    annotationLoadStateRef.current[currentEntry.id] = 'ready'

    setAnnotationsByImage((current) => {
      const currentAnnotations = current[currentEntry.id] ?? []
      const sourceIndex = currentAnnotations.findIndex(
        (annotation) => annotation.id === annotationId,
      )
      if (sourceIndex < 0) {
        return current
      }

      const boundedInsertIndex = Math.max(
        0,
        Math.min(insertIndex, currentAnnotations.length),
      )
      if (
        boundedInsertIndex === sourceIndex ||
        boundedInsertIndex === sourceIndex + 1
      ) {
        return current
      }

      const nextAnnotations = [...currentAnnotations]
      const [movedAnnotation] = nextAnnotations.splice(sourceIndex, 1)
      const adjustedInsertIndex =
        sourceIndex < boundedInsertIndex
          ? boundedInsertIndex - 1
          : boundedInsertIndex
      nextAnnotations.splice(adjustedInsertIndex, 0, movedAnnotation)

      return {
        ...current,
        [currentEntry.id]: nextAnnotations,
      }
    })
  }

  const clearCurrentImageAnnotations = () => {
    if (!currentEntry) {
      return
    }

    annotationLoadStateRef.current[currentEntry.id] = 'ready'

    setAnnotationsByImage((current) => ({
      ...current,
      [currentEntry.id]: [],
    }))
    setSelectedId(null)
  }

  const onGlobalKeyDown = useEffectEvent((event: KeyboardEvent) => {
    if (
      hotkeyCaptureTarget ||
      confirmDialogState ||
      isClassManagerOpen ||
      isHotkeysOpen
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

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
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
      imageIdSetRef.current = new Set(nextImages.map((entry) => entry.id))

      if (
        preferredImageRelativePath &&
        preferredImageIndex >= 0 &&
        pendingPreferredImageRelativePathRef.current === preferredImageRelativePath
      ) {
        pendingPreferredImageRelativePathRef.current = null
        setCurrentImageEntry(nextImages[preferredImageIndex] ?? null)
      }

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
        setSessionLabel(nextSessionLabel)
        setSessionError(null)
      })
      return
    }

    sessionVersionRef.current += 1
    imageIdSetRef.current = new Set(nextImages.map((entry) => entry.id))
    pendingLoadsRef.current = {}
    pendingAnnotationLoadsRef.current = {}
    annotationLoadStateRef.current = {}
    imageResourcesRef.current = {}
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
      setAnnotationsByImage({})
      setCurrentSessionId(sessionId)
      setCurrentImageEntry(initialEntry)
      setSessionLabel(nextSessionLabel)
      setSessionLoadProgress(null)
      setSessionError(null)
      setSelectedId(null)
      setDrawStart(null)
      setDraftRect(null)
      setCustomClasses([])
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
    setSelectedId(null)
    setDrawStart(point)
    setDraftRect({ x: point.x, y: point.y, width: 0, height: 0 })
  })

  const handleCanvasPointerMove = useEffectEvent((point: Point) => {
    if (!drawStart) {
      return
    }

    setDraftRect(rectFromPoints(drawStart, point))
  })

  const handleCanvasPointerUp = useEffectEvent((point: Point) => {
    if (!drawStart || !image) {
      return
    }

    const nextRect = rectFromPoints(drawStart, point)
    setDrawStart(null)
    setDraftRect(null)

    if (nextRect.width < MIN_BOX_SIZE || nextRect.height < MIN_BOX_SIZE) {
      return
    }

    const label = activeLabel.trim() || 'object'
    const nextAnnotation: Annotation = {
      id: crypto.randomUUID(),
      label,
      color: labelToColor(label),
      difficult: false,
      ...nextRect,
    }

    setAnnotationsByImage((current) => ({
      ...current,
      [image.id]: [...(current[image.id] ?? []), nextAnnotation],
    }))
    annotationLoadStateRef.current[image.id] = 'ready'
    setSelectedId(nextAnnotation.id)
  })

  const handleLabelExport = (format: 'json' | 'voc' | 'yolo') => {
    if (!image || annotations.length === 0) {
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

    const yolo = serializeYolo(image, annotations)
    downloadTextFile(`${baseName}.txt`, yolo.annotationText)
    downloadTextFile('classes.txt', yolo.classesText)
  }

  return (
    <div className="app-shell">
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
                  disabled={!image || annotations.length === 0}
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
              onOpenDataset={handleOpenLocalDirectory}
              recentDatasets={recentDatasets}
              onOpenRecentDataset={handleOpenRecentDataset}
              onRemoveRecentDataset={removeRecentDataset}
              openDatasetDisabled={isOpeningSession || backendStatus !== 'online'}
              onStartDrawing={handleCanvasPointerDown}
              onUpdateDrawing={handleCanvasPointerMove}
              onFinishDrawing={handleCanvasPointerUp}
              onSelectAnnotation={setSelectedId}
              onUpdateAnnotationRect={updateAnnotationRect}
              onDuplicateAnnotation={duplicateAnnotation}
              onDeleteAnnotation={removeAnnotation}
            />
          </div>
        </main>

        {!isSidebarVisible ? (
          <AppButton
            variant="menu-trigger"
            className="workspace-sidebar-reveal"
            onClick={() => setIsSidebarVisible(true)}
            aria-label="Show sidebar"
            title="Show sidebar"
          >
            Open Menu
          </AppButton>
        ) : null}

        <aside
          className={
            isSidebarVisible ? 'panel panel-sidebar' : 'panel panel-sidebar is-hidden'
          }
        >
          <div className="panel-sidebar-toggle-row">
            <AppButton
              variant="menu-trigger"
              className="panel-sidebar-toggle"
              onClick={() => setIsSidebarVisible(false)}
              aria-label="Hide sidebar"
              title="Hide sidebar"
            >
              {'>'}
            </AppButton>
          </div>

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
                    {images.length > 0 ? `${currentImageIndex + 1}/${images.length}` : '0/0'}
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

            {sessionError ? <p className="path-note">{sessionError}</p> : null}

            {images.length > 0 ? (
              <VirtualFileList
                images={images}
                currentIndex={currentImageIndex}
                onSelectIndex={selectImageIndex}
              />
            ) : null}
          </section>

          <section className="panel-section">
            <div className="section-heading-row section-heading-row-kicker">
              <p className="section-kicker section-kicker-inline">Labeling</p>
              <AppButton
                variant="ghost"
                className="class-add-button"
                onClick={openClassManager}
              >
                Manage
              </AppButton>
            </div>
            <div className="chip-list" aria-label="Known classes">
              {classList.length > 0 ? (
                classList.map((label) => (
                  <AppButton
                    key={label}
                    variant="chip"
                    isActive={label === activeLabel}
                    onClick={() => setActiveLabel(label)}
                  >
                    {label}
                  </AppButton>
                ))
              ) : (
                <span className="muted">Classes appear here after annotation.</span>
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
                        Dup
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
                        Del
                      </AppButton>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <span className="muted">No boxes on this image.</span>
            )}
          </section>
        </aside>
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
            aria-label="Manage classes"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="class-manager-header">
              <div>
                <p className="section-kicker">Classes</p>
                <h2>Manage classes</h2>
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
                placeholder="New class name"
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
                  const isPreset = backendClasses.includes(label)
                  const usageCount = classUsageCounts[label] ?? 0
                  const isEditing = editingClassLabel === label
                  const canDelete = !isPreset && usageCount === 0

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
                            label === activeLabel
                              ? 'class-manager-main is-active'
                              : 'class-manager-main'
                          }
                          onClick={() => setActiveLabel(label)}
                        >
                          <span className="class-manager-name">{label}</span>
                          <span className="class-manager-meta">
                            {isPreset ? 'preset' : 'custom'}
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
                              className="class-manager-action"
                              onClick={() => startEditingClass(label)}
                              disabled={isPreset}
                              title={
                                isPreset
                                  ? 'Preset classes cannot be renamed here'
                                  : 'Rename class'
                              }
                            >
                              Rename
                            </AppButton>
                            <AppButton
                              variant="ghost"
                              className="class-manager-action is-danger"
                              onClick={() => removeClass(label)}
                              disabled={!canDelete}
                              title={
                                isPreset
                                  ? 'Preset classes cannot be deleted here'
                                  : usageCount > 0
                                    ? 'Reassign or delete boxes before removing this class'
                                    : 'Delete class'
                              }
                            >
                              Delete
                            </AppButton>
                          </>
                        )}
                      </div>
                    </div>
                  )
                })
              ) : (
                <span className="muted">No classes available.</span>
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
                                      className="hotkeys-binding-action"
                                      onClick={() =>
                                        beginHotkeyCapture(item.id, index)
                                      }
                                    >
                                      {isCapturing &&
                                      hotkeyCaptureTarget?.bindingIndex === index
                                        ? 'Press keys…'
                                        : 'Edit'}
                                    </AppButton>
                                    <AppButton
                                      variant="ghost"
                                      className="hotkeys-binding-action is-danger"
                                      onClick={() =>
                                        removeHotkeyBinding(item.id, index)
                                      }
                                    >
                                      Remove
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
                                className="hotkeys-binding-action"
                                onClick={() => beginHotkeyCapture(item.id, null)}
                              >
                                {isCapturing &&
                                hotkeyCaptureTarget?.bindingIndex === null
                                  ? 'Press keys…'
                                  : 'Add shortcut'}
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
