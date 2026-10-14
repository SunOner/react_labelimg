import { useEffect, useEffectEvent, useRef, useState } from 'react'
import { AnnotationViewport } from './AnnotationViewport'
import { VideoSourceHistory } from './VideoSourceHistory'
import { AppButton } from './ui/AppButton'
import { HistoryButtons } from './ui/HistoryButtons'
import { AnnotationHistory, annotationsEqual, type HistoryAvailability } from '../lib/annotationHistory'
import {
  buildOriginalVideoUrl,
  fetchDatasetVideos,
  fetchLocalAnnotations,
  openOriginalVideoFile,
  openOriginalVideoPath,
  registerBrowserVideo,
  saveVideoProgress,
  type LocalSessionImage,
  type VideoMetadata,
  type VideoProgress,
} from '../lib/api'
import {
  labelToColor,
  MIN_ANNOTATION_SIZE,
  rectFromPoints,
} from '../lib/annotations'
import {
  boundedVideoTime,
  captureVideoFrame,
  formatVideoTime,
  identifyVideo,
  videoFrameFilename,
  videoFrameKey,
} from '../lib/video'
import type { Annotation, LoadedImage, Point, Rect } from '../types'
import './VideoWorkbench.css'

type Frame = {
  image: LoadedImage
  blob: Blob
  time: number
  filename: string
  savedImage: LocalSessionImage | null
}
type SavedFrame = { time: number; image: LocalSessionImage }
type Props = {
  sessionId: string
  datasetLabel: string
  classes: string[]
  datasetImages: LocalSessionImage[]
  initialLabel: string
  initialFile?: File | null
  undoShortcut?: string
  redoShortcut?: string
  matchesUndoHotkey?: (event: KeyboardEvent) => boolean
  matchesRedoHotkey?: (event: KeyboardEvent) => boolean
  onSave: (
    file: File,
    annotations: Annotation[],
    existing: LocalSessionImage | null,
    source: { videoId: string; timeSeconds: number },
  ) => Promise<{ image: LocalSessionImage; annotations: Annotation[] }>
  onClose: () => void
}

export function VideoWorkbench({
  sessionId,
  datasetLabel,
  classes,
  datasetImages,
  initialLabel,
  initialFile,
  undoShortcut,
  redoShortcut,
  matchesUndoHotkey,
  matchesRedoHotkey,
  onSave,
  onClose,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  const mediaTimeRef = useRef<number | null>(null)
  const presentedSinceSeekRef = useRef(false)
  const captureTimerRef = useRef<number | null>(null)
  const pendingCaptureRef = useRef(false)
  const latestCaptureRef = useRef<(() => Promise<void>) | null>(null)
  const dirtyRef = useRef(false)
  const generationRef = useRef(0)
  const operationRef = useRef(false)
  const frameUrlRef = useRef<string | null>(null)
  const drawStartRef = useRef<Point | null>(null)
  const hoverPointRef = useRef<Point | null>(null)
  const baselineAnnotationsRef = useRef<Annotation[]>([])
  const annotationsRef = useRef<Annotation[]>([])
  const progressRequestRef = useRef<Promise<boolean> | null>(null)
  const lastProgressRef = useRef<string | null>(null)
  const pendingActionRef = useRef<(() => void) | null>(null)
  const [lastBoxes, setLastBoxes] = useState<Annotation[]>([])
  const [source, setSource] = useState<{
    name: string
    url: string
    id: string
    metadata: VideoMetadata
    local: boolean
    token: number
  } | null>(null)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [annotationHistory] = useState(() => new AnnotationHistory(50, 1))
  const [historyAvailability, setHistoryAvailability] = useState<HistoryAvailability>({ canUndo: false, canRedo: false })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draftRect, setDraftRect] = useState<Rect | null>(null)
  const [label, setLabel] = useState(initialLabel || classes[0] || 'object')
  const [tool, setTool] = useState<'draw' | 'new-box'>('draw')
  const [crosshair, setCrosshair] = useState(true)
  const [duration, setDuration] = useState(0)
  const [time, setTime] = useState(0)
  const [timelineDraft, setTimelineDraft] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [seeking, setSeeking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [confirmNavigation, setConfirmNavigation] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metadataError, setMetadataError] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [metadataReady, setMetadataReady] = useState(false)
  const [videoHistory, setVideoHistory] = useState<VideoMetadata[]>([])
  const [message, setMessage] = useState('')
  const [fps, setFps] = useState('30')
  const [interval, setInterval] = useState('1')
  const [jumpTime, setJumpTime] = useState('0')
  const [speed, setSpeed] = useState('1')
  const [format, setFormat] = useState<'png' | 'jpg'>('png')
  const [savedFrames, setSavedFrames] = useState<Record<string, SavedFrame>>({})
  const frameRate = Math.max(1, Math.min(240, Number(fps) || 30))
  const sampleInterval = Math.max(0.001, Number(interval) || 1)
  const selected = annotations.find((box) => box.id === selectedId)
  const datasetName = datasetLabel.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || datasetLabel
  const classOptions = [
    ...new Set(
      [...classes, label.trim(), ...annotations.map((box) => box.label)].filter(
        Boolean,
      ),
    ),
  ]

  const releaseFrame = () => {
    if (captureTimerRef.current !== null)
      window.clearTimeout(captureTimerRef.current)
    captureTimerRef.current = null
    pendingCaptureRef.current = false
    if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
    frameUrlRef.current = null
    setFrame(null)
    setAnnotations([])
    annotationsRef.current = []
    annotationHistory.clear()
    setHistoryAvailability({ canUndo: false, canRedo: false })
    baselineAnnotationsRef.current = []
    setSelectedId(null)
    drawStartRef.current = null
    hoverPointRef.current = null
    setDraftRect(null)
    dirtyRef.current = false
    setDirty(false)
  }

  const rememberVideo = (video: VideoMetadata) =>
    setVideoHistory((current) => [
      video,
      ...current.filter((entry) => entry.id !== video.id),
    ])

  const activateVideo = (
    metadata: VideoMetadata,
    url: string,
    local: boolean,
  ) => {
    const id = metadata.id
    videoRef.current?.pause()
    releaseFrame()
    setMetadataReady(false)
    lastProgressRef.current = null
    setMetadataError(null)
    setSource({
      name: metadata.name,
      url,
      id,
      metadata,
      local,
      token: generationRef.current,
    })
    setDuration(0)
    setTime(0)
    setPlaying(false)
    setSeeking(false)
    setFps(String(metadata.state.stepFps))
    setInterval(String(metadata.state.sampleIntervalSeconds))
    setFormat(metadata.state.imageFormat)
    setSpeed(String(metadata.state.playbackRate))
    const existingFrames: Record<string, SavedFrame> = {}
    const pattern = new RegExp(`_${id}_t(\\d+)us\\.(png|jpe?g)$`, 'i')
    for (const image of datasetImages) {
      const match = image.name.match(pattern)
      if (match) {
        const timestamp = Number(match[1]) / 1_000_000
        existingFrames[videoFrameKey(timestamp)] = { time: timestamp, image }
      }
    }
    setSavedFrames(existingFrames)
    setMessage('')
    setLastBoxes([])
    mediaTimeRef.current = null
    presentedSinceSeekRef.current = false
    rememberVideo(metadata)
  }

  const progressSnapshot = (): VideoProgress => ({
    positionSeconds: videoRef.current?.currentTime ?? time,
    durationSeconds: duration,
    width: videoRef.current?.videoWidth ?? 0,
    height: videoRef.current?.videoHeight ?? 0,
    stepFps: frameRate,
    sampleIntervalSeconds: Math.min(86400, sampleInterval),
    imageFormat: format,
    playbackRate: Number(speed),
  })

  async function persistProgress(force = false): Promise<boolean> {
    if (!source || !metadataReady) return true
    if (progressRequestRef.current) {
      if (!force) return progressRequestRef.current
      await progressRequestRef.current
    }
    const state = progressSnapshot()
    if (!state.width || !state.height || !state.durationSeconds) return true
    const key = JSON.stringify([source.id, state])
    if (key === lastProgressRef.current) return true
    const request = saveVideoProgress(sessionId, source.id, state)
      .then((video) => {
        lastProgressRef.current = key
        rememberVideo(video)
        setMetadataError(null)
        return true
      })
      .catch((cause: unknown) => {
        setMetadataError(
          cause instanceof Error
            ? cause.message
            : 'Could not save video progress.',
        )
        return false
      })
      .finally(() => {
        progressRequestRef.current = null
      })
    progressRequestRef.current = request
    return request
  }

  const persistCurrentProgress = useEffectEvent(() => {
    void persistProgress()
  })
  const persistOnPageHide = useEffectEvent(() => {
    if (source && metadataReady)
      void saveVideoProgress(
        sessionId,
        source.id,
        progressSnapshot(),
        true,
      ).catch(() => {})
  })
  useEffect(() => {
    if (!metadataReady) return
    const timer = window.setInterval(() => persistCurrentProgress(), 2000)
    const onHide = () => persistOnPageHide()
    window.addEventListener('pagehide', onHide)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('pagehide', onHide)
    }
  }, [metadataReady, source?.id])
  useEffect(() => {
    if (!metadataReady) return
    const timer = window.setTimeout(() => persistCurrentProgress(), 500)
    return () => window.clearTimeout(timer)
  }, [metadataReady, source?.id, time, fps, interval, format, speed])

  const refreshHistory = async (signal?: AbortSignal) => {
    try {
      setVideoHistory(await fetchDatasetVideos(sessionId, signal))
      setHistoryError(null)
    } catch (cause) {
      if (!signal?.aborted)
        setHistoryError(
          cause instanceof Error
            ? cause.message
            : 'Could not load video history.',
        )
    }
  }
  const refreshInitialHistory = useEffectEvent((signal: AbortSignal) => {
    void refreshHistory(signal)
  })
  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(
      () => refreshInitialHistory(controller.signal),
      0,
    )
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [sessionId])

  const loadFile = async (file: File) => {
    if (operationRef.current) return
    operationRef.current = true
    const generation = ++generationRef.current
    setBusy(true)
    setError(null)
    try {
      if (!(await persistProgress(true))) return
      const id = await identifyVideo(file)
      const metadata = await registerBrowserVideo(sessionId, file, id)
      if (generation !== generationRef.current) return
      activateVideo(metadata, URL.createObjectURL(file), true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load video.')
    } finally {
      operationRef.current = false
      setBusy(false)
    }
  }

  const loadOriginal = async (path?: string) => {
    if (operationRef.current) return
    operationRef.current = true
    const generation = ++generationRef.current
    setBusy(true)
    setError(null)
    try {
      if (!(await persistProgress(true))) return
      const result = path
        ? {
            cancelled: false,
            video: await openOriginalVideoPath(sessionId, path),
          }
        : await openOriginalVideoFile(sessionId)
      if (result.video && generation === generationRef.current)
        activateVideo(
          result.video,
          buildOriginalVideoUrl(sessionId, result.video.id),
          false,
        )
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not open original video.',
      )
    } finally {
      operationRef.current = false
      setBusy(false)
    }
  }

  const closeWorkbench = async () => {
    videoRef.current?.pause()
    if (await persistProgress(true)) onClose()
  }

  const loadInitialFile = useEffectEvent(() => {
    if (initialFile) void loadFile(initialFile)
  })

  useEffect(() => {
    const timer = window.setTimeout(() => loadInitialFile(), 0)
    return () => window.clearTimeout(timer)
  }, [initialFile])

  const capturePresentedFrame = useEffectEvent(() => {
    const video = videoRef.current
    if (!pendingCaptureRef.current || !video?.paused || video.seeking) return
    if (captureTimerRef.current !== null)
      window.clearTimeout(captureTimerRef.current)
    captureTimerRef.current = null
    pendingCaptureRef.current = false
    void capture()
  })

  useEffect(() => {
    if (!source) return
    const video = videoRef.current
    let callbackId: number | null = null
    let cancelled = false
    if (video?.requestVideoFrameCallback) {
      const track: VideoFrameRequestCallback = (_now, metadata) => {
        if (cancelled) return
        mediaTimeRef.current = metadata.mediaTime
        presentedSinceSeekRef.current = true
        capturePresentedFrame()
        callbackId = video.requestVideoFrameCallback(track)
      }
      callbackId = video.requestVideoFrameCallback(track)
    }
    return () => {
      cancelled = true
      if (video && callbackId !== null)
        video.cancelVideoFrameCallback(callbackId)
      if (source.local) URL.revokeObjectURL(source.url)
    }
  }, [source])

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null
    dialogRef.current?.focus()
    return () => {
      generationRef.current += 1
      if (captureTimerRef.current !== null)
        window.clearTimeout(captureTimerRef.current)
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      previousFocus?.focus()
    }
  }, [])

  useEffect(() => {
    if (!dirty && !busy) return
    const beforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [dirty, busy])

  useEffect(() => {
    if (!confirmNavigation) return
    const previous = document.activeElement as HTMLElement | null
    dialogRef.current
      ?.querySelector<HTMLElement>('.video-confirm button')
      ?.focus()
    return () => previous?.focus()
  }, [confirmNavigation])

  async function capture() {
    const video = videoRef.current
    if (
      !video ||
      !source ||
      operationRef.current ||
      video.seeking ||
      !video.paused ||
      dirtyRef.current
    )
      return
    const generation = generationRef.current
    operationRef.current = true
    setBusy(true)
    setError(null)
    try {
      const timestamp = mediaTimeRef.current ?? video.currentTime
      const width = video.videoWidth
      const height = video.videoHeight
      const blob = await captureVideoFrame(video, format)
      const saved = savedFrames[videoFrameKey(timestamp)]?.image ?? null
      const boxes = saved
        ? (await fetchLocalAnnotations(sessionId, saved.id)).annotations.map(
            (box) => ({ ...box, color: labelToColor(box.label) }),
          )
        : []
      if (generation !== generationRef.current) return
      releaseFrame()
      const url = URL.createObjectURL(blob)
      frameUrlRef.current = url
      const filename = videoFrameFilename(
        source.name,
        source.id,
        timestamp,
        format,
      )
      setFrame({
        blob,
        time: timestamp,
        filename,
        savedImage: saved,
        image: {
          id: videoFrameKey(timestamp),
          name: filename,
          relativePath: filename,
          url,
          width,
          height,
          annotationCount: boxes.length,
        },
      })
      setTime(video.currentTime)
      setAnnotations(boxes)
      annotationsRef.current = boxes
      annotationHistory.clear()
      setHistoryAvailability({ canUndo: false, canRedo: false })
      baselineAnnotationsRef.current = boxes
      setMessage(
        saved
          ? 'Saved frame — edits update its annotations.'
          : 'Draw boxes on this frame, then save it to the dataset.',
      )
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'Could not capture frame.',
      )
    } finally {
      operationRef.current = false
      setBusy(false)
    }
  }

  useEffect(() => {
    latestCaptureRef.current = capture
  })

  const queueCapture = () => {
    const video = videoRef.current
    if (
      !video ||
      !source ||
      operationRef.current ||
      dirtyRef.current ||
      video.seeking ||
      !video.paused
    )
      return
    if (captureTimerRef.current !== null)
      window.clearTimeout(captureTimerRef.current)
    if (
      !video.requestVideoFrameCallback ||
      (presentedSinceSeekRef.current && mediaTimeRef.current !== null)
    ) {
      void capture()
      return
    }
    // seeked can precede frame presentation. Wait for its timestamp rather than
    // assigning the requested seek time to a different decoded frame.
    pendingCaptureRef.current = true
    setBusy(true)
    const generation = generationRef.current
    captureTimerRef.current = window.setTimeout(() => {
      captureTimerRef.current = null
      pendingCaptureRef.current = false
      if (generation === generationRef.current)
        void latestCaptureRef.current?.()
    }, 300)
  }

  const save = async (mode: 'annotations' | 'background' = 'annotations') => {
    if (!frame || !source || operationRef.current) return false
    if (drawStartRef.current) {
      setError('Finish drawing the current box before saving.')
      return false
    }
    if (annotations.length === 0 && mode !== 'background') {
      setError('Add annotations before saving, or choose Save as background for an empty frame.')
      return false
    }
    if (annotations.length > 0 && mode === 'background') {
      setError('A background frame must have no annotations. Use the regular save button for this frame.')
      return false
    }
    if (annotations.some((box) => !box.label.trim())) {
      setError('Give every box a class before saving.')
      return false
    }
    operationRef.current = true
    setBusy(true)
    setError(null)
    try {
      const file = new File([frame.blob], frame.filename, {
        type: frame.blob.type,
      })
      const { image, annotations: savedBoxes } = await onSave(
        file,
        annotations.map((box) => ({ ...box, label: box.label.trim() })),
        frame.savedImage,
        { videoId: source.id, timeSeconds: frame.time },
      )
      setFrame({ ...frame, savedImage: image })
      setSavedFrames((current) => ({
        ...current,
        [videoFrameKey(frame.time)]: { time: frame.time, image },
      }))
      setAnnotations(savedBoxes)
      annotationsRef.current = savedBoxes
      baselineAnnotationsRef.current = savedBoxes
      setLastBoxes(savedBoxes.map((box) => ({ ...box })))
      dirtyRef.current = false
      setDirty(false)
      setMessage(mode === 'background'
        ? `Saved as background: ${image.name}`
        : `Saved ${image.name} · ${savedBoxes.length} boxes`)
      return true
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not save frame. Your annotations are still here.',
      )
      return false
    } finally {
      operationRef.current = false
      setBusy(false)
    }
  }

  const navigate = (action: () => void) => {
    if (operationRef.current) return
    if (dirty) {
      pendingActionRef.current = action
      setConfirmNavigation(true)
    } else action()
  }

  const commitTimeline = (value: string) => {
    setTimelineDraft(null)
    const target = Number(value)
    if (Number.isFinite(target)) navigate(() => seek(target))
  }

  const seek = (target: number) => {
    const video = videoRef.current
    if (!video || !duration) return
    video.pause()
    releaseFrame()
    setError(null)
    const next = boundedVideoTime(target, duration)
    if (Math.abs(next - video.currentTime) < 0.000001) {
      const generation = generationRef.current
      captureTimerRef.current = window.setTimeout(() => {
        captureTimerRef.current = null
        if (generation === generationRef.current)
          void latestCaptureRef.current?.()
      }, 0)
      return
    }
    presentedSinceSeekRef.current = false
    setSeeking(true)
    video.currentTime = next
    setTime(next)
  }

  const togglePlayback = () =>
    navigate(() => {
      const video = videoRef.current
      if (!video) return
      if (!video.paused) video.pause()
      else {
        releaseFrame()
        void video
          .play()
          .catch((cause: unknown) =>
            setError(
              cause instanceof Error ? cause.message : 'Could not play video.',
            ),
          )
      }
    })

  const edit = (next: Annotation[], nextSelectedId: string | null = selectedId) => {
    if (operationRef.current) return
    annotationHistory.record('frame', { annotations: annotationsRef.current, selectedId }, {
      annotations: next,
      selectedId: nextSelectedId,
    })
    annotationsRef.current = next
    setAnnotations(next)
    setSelectedId(nextSelectedId)
    const changed = !annotationsEqual(next, baselineAnnotationsRef.current)
    dirtyRef.current = changed
    setDirty(changed)
    setHistoryAvailability(annotationHistory.state('frame'))
    setMessage(changed ? 'Unsaved annotations' : 'Annotations restored')
  }

  const beginAnnotationInteraction = () => {
    annotationHistory.beginGroup('frame')
    setHistoryAvailability(annotationHistory.state('frame'))
  }
  const endAnnotationInteraction = () => {
    annotationHistory.endGroup()
    setHistoryAvailability(annotationHistory.state('frame'))
  }
  const restoreAnnotationHistory = (direction: 'undo' | 'redo') => {
    if (!frame || operationRef.current || seeking) return
    if (drawStartRef.current) {
      drawStartRef.current = null
      setDraftRect(null)
      setTool('draw')
      return
    }
    const snapshot = annotationHistory[direction]('frame')
    if (!snapshot) return
    annotationsRef.current = snapshot.annotations
    setAnnotations(snapshot.annotations)
    setSelectedId(snapshot.selectedId)
    const changed = !annotationsEqual(snapshot.annotations, baselineAnnotationsRef.current)
    dirtyRef.current = changed
    setDirty(changed)
    setHistoryAvailability(annotationHistory.state('frame'))
    setError(null)
    setMessage(changed ? 'Unsaved annotations' : 'Annotations restored')
  }
  const updateBox = (id: string, patch: Partial<Annotation>) =>
    edit(annotations.map((box) => (box.id === id ? { ...box, ...patch } : box)))
  const deleteBox = (id: string) => {
    edit(annotations.filter((box) => box.id !== id), null)
  }
  const duplicateBox = (id: string) => {
    const box = annotations.find((item) => item.id === id)
    if (!box) return
    const copy = { ...box, id: crypto.randomUUID() }
    edit([...annotations, copy], copy.id)
  }
  const finishDrawing = (point: Point) => {
    const start = drawStartRef.current
    drawStartRef.current = null
    setDraftRect(null)
    setTool('draw')
    if (!start || operationRef.current) return
    const rect = rectFromPoints(start, point)
    if (rect.width < MIN_ANNOTATION_SIZE || rect.height < MIN_ANNOTATION_SIZE)
      return
    const name = label.trim() || 'object'
    const box: Annotation = {
      ...rect,
      id: crypto.randomUUID(),
      label: name,
      color: labelToColor(name),
      difficult: false,
    }
    edit([...annotations, box], box.id)
  }

  const onKeyDown = useEffectEvent((event: KeyboardEvent) => {
    if (event.key === 'Tab') {
      const controls = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), summary, [tabindex="0"]',
        ) ?? [],
      ).filter(
        (node) => node.getClientRects().length > 0 && !node.closest('[inert]'),
      )
      const first = controls[0],
        last = controls.at(-1)
      if (
        event.shiftKey &&
        (document.activeElement === first ||
          document.activeElement === dialogRef.current)
      ) {
        event.preventDefault()
        last?.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first?.focus()
      }
      return
    }
    if (event.key === 'Escape') {
      event.preventDefault()
      if (drawStartRef.current) {
        drawStartRef.current = null
        setDraftRect(null)
        setTool('draw')
      } else if (confirmNavigation) {
        setConfirmNavigation(false)
        pendingActionRef.current = null
      } else navigate(() => void closeWorkbench())
      return
    }
    if (confirmNavigation || busy) return
    const target = event.target as HTMLElement
    const editable =
      ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName) ||
      target.isContentEditable
    const undoRequested = matchesUndoHotkey ? matchesUndoHotkey(event)
      : !event.altKey && (event.ctrlKey || event.metaKey) && event.code === 'KeyZ' && !event.shiftKey
    const redoRequested = matchesRedoHotkey ? matchesRedoHotkey(event)
      : !event.altKey && (event.ctrlKey || event.metaKey) && (event.code === 'KeyY' || (event.code === 'KeyZ' && event.shiftKey))
    if (!editable && (undoRequested || redoRequested)) {
      event.preventDefault()
      restoreAnnotationHistory(undoRequested ? 'undo' : 'redo')
      return
    }
    if ((event.ctrlKey || event.metaKey) && event.code === 'KeyS') {
      event.preventDefault()
      void save()
      return
    }
    if (
      editable ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      event.repeat
    )
      return
    if (event.code === 'Space' && target.tagName !== 'BUTTON') {
      event.preventDefault()
      togglePlayback()
    }
    if (event.code === 'KeyW' && frame) {
      event.preventDefault()
      const start = hoverPointRef.current ?? {
        x: frame.image.width / 2,
        y: frame.image.height / 2,
      }
      drawStartRef.current = start
      setDraftRect({ ...start, width: 0, height: 0 })
      setSelectedId(null)
      setTool('new-box')
    }
    if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
      event.preventDefault()
      navigate(() =>
        seek(
          time +
            (event.code === 'ArrowLeft' ? -1 : 1) *
              (event.shiftKey ? sampleInterval : 1 / frameRate),
        ),
      )
    }
    if ((event.code === 'Delete' || event.code === 'Backspace') && selectedId) {
      event.preventDefault()
      deleteBox(selectedId)
    }
    if (/^Digit[1-9]$/.test(event.code)) {
      const next = classOptions[Number(event.code.slice(-1)) - 1]
      if (next) {
        event.preventDefault()
        setLabel(next)
        if (selectedId)
          updateBox(selectedId, {
            label: next,
            color: labelToColor(next),
            sourceClassIndex: null,
            hasUnknownClass: false,
          })
      }
    }
  })
  useEffect(() => {
    const handler = (event: KeyboardEvent) => onKeyDown(event)
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const continueNavigation = async (saveFirst: boolean, mode: 'annotations' | 'background' = 'annotations') => {
    if (saveFirst && !(await save(mode))) return
    if (!saveFirst) {
      setAnnotations(baselineAnnotationsRef.current)
      annotationsRef.current = baselineAnnotationsRef.current
      annotationHistory.clear()
      setHistoryAvailability({ canUndo: false, canRedo: false })
      setSelectedId(null)
      setMessage('Changes discarded')
    }
    const action = pendingActionRef.current
    pendingActionRef.current = null
    setConfirmNavigation(false)
    dirtyRef.current = false
    setDirty(false)
    action?.()
  }

  return (
    <div
      className="video-workbench-backdrop"
      onDragOver={(event) => {
        event.preventDefault()
        event.stopPropagation()
      }}
      onDragEnter={(event) => event.stopPropagation()}
      onDrop={(event) => {
        event.preventDefault()
        event.stopPropagation()
        const file = event.dataTransfer.files[0]
        if (file) navigate(() => void loadFile(file))
      }}
    >
      <div
        className="video-workbench"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Video annotation workspace"
        tabIndex={-1}
      >
        <header className="video-header" inert={confirmNavigation}>
          <div className="video-heading">
            <div className="video-title-row">
              <h2>Video annotation</h2>
              <span className="video-dataset" title={datasetLabel}>Dataset: <strong>{datasetName}</strong></span>
            </div>
            {source ? (
              <p title={source.metadata.sourcePath ?? source.name}>
                {source.name}
              </p>
            ) : null}
          </div>
          <div className="video-header-actions">
            <HistoryButtons {...historyAvailability} disabled={!frame || busy || seeking}
              undoShortcut={undoShortcut} redoShortcut={redoShortcut}
              onUndo={() => restoreAnnotationHistory('undo')} onRedo={() => restoreAnnotationHistory('redo')} />
            <VideoSourceHistory
              videos={videoHistory}
              error={historyError}
              busy={busy}
              currentId={source?.id}
              currentFrameCount={Object.keys(savedFrames).length}
              onRefresh={() => void refreshHistory()}
              onChooseBrowserFile={() => fileInputRef.current?.click()}
              onChooseOriginalFile={() => navigate(() => void loadOriginal())}
              onOpenPath={(path) => navigate(() => void loadOriginal(path))}
            />
            <AppButton
              variant="primary"
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              {source ? 'Change video' : 'Open video'}
            </AppButton>
            <AppButton
              className="video-close"
              aria-label="Close"
              title="Close (Esc)"
              disabled={busy}
              onClick={() => navigate(() => void closeWorkbench())}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6" /></svg>
            </AppButton>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,.mp4,.webm,.mov,.m4v,.mkv,.avi"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0]
              event.target.value = ''
              if (file) navigate(() => void loadFile(file))
            }}
          />
        </header>
        {metadataError ? (
          <div className="video-error" role="alert">
            <span>
              Video metadata: {metadataError}. Dataset frames are unaffected.
            </span>
            <div className="video-actions">
              <AppButton
                disabled={busy}
                onClick={() => void persistProgress(true)}
              >
                Retry saving progress
              </AppButton>
              <AppButton disabled={busy} onClick={() => navigate(onClose)}>
                Close without saving progress
              </AppButton>
            </div>
          </div>
        ) : null}
        {error ? (
          <div className="video-error" role="alert">
            {error}
          </div>
        ) : null}
        <div className="video-body" inert={confirmNavigation}>
          <div className="video-stage">
            <video
              key={source?.token ?? 'no-source'}
              ref={videoRef}
              src={source?.url}
              crossOrigin={source?.local ? undefined : 'anonymous'}
              playsInline
              muted
              preload="auto"
              hidden={!source}
              aria-hidden={!source || !!frame}
              className={
                !source || frame ? 'video-player is-covered' : 'video-player'
              }
              onLoadedMetadata={(event) => {
                const video = event.currentTarget
                if (!Number.isFinite(video.duration) || video.duration <= 0) {
                  setError(
                    'This video has no usable duration. Try another file.',
                  )
                  return
                }
                setDuration(video.duration)
                video.playbackRate = Number(speed)
                const resume = boundedVideoTime(
                  source?.metadata.state.positionSeconds ?? 0,
                  video.duration,
                )
                setTime(resume)
                if (resume > 0) {
                  presentedSinceSeekRef.current = false
                  video.currentTime = resume
                }
                setMetadataReady(true)
              }}
              onLoadedData={queueCapture}
              onTimeUpdate={(event) => {
                if (!frame) setTime(event.currentTarget.currentTime)
              }}
              onPlay={() => setPlaying(true)}
              onPause={() => {
                setPlaying(false)
                if (!videoRef.current?.seeking) queueCapture()
              }}
              onSeeking={() => {
                setSeeking(true)
                presentedSinceSeekRef.current = false
              }}
              onSeeked={() => {
                setSeeking(false)
                queueCapture()
              }}
              onEnded={() => {
                setPlaying(false)
                queueCapture()
              }}
              onError={() => {
                setPlaying(false)
                setSeeking(false)
                setDuration(0)
                setError(
                  'The browser could not decode this video. Try MP4 (H.264) or WebM, or convert the source video to one of these formats.',
                )
              }}
            />
            {frame ? (
              <AnnotationViewport
                key={`${source?.id}-${frame.image.id}`}
                image={frame.image}
                imageLabel={`${source?.name} · ${formatVideoTime(frame.time)}`}
                isLoading={false}
                annotations={annotations}
                selectedId={selectedId}
                draftRect={draftRect}
                tool={tool}
                isCrosshairEnabled={crosshair}
                showSamTools={false}
                isSamBusy={busy}
                classOptions={classOptions}
                onSelectTool={(next) => {
                  if (next === 'draw' || next === 'new-box') {
                    setTool(next)
                    drawStartRef.current = null
                    setDraftRect(null)
                  }
                }}
                onToggleCrosshair={() => setCrosshair((current) => !current)}
                onOpenDataset={() => {}}
                recentDatasets={[]}
                onOpenRecentDataset={() => {}}
                onRemoveRecentDataset={() => {}}
                onStartDrawing={(point) => {
                  if (busy) return
                  drawStartRef.current = point
                  setDraftRect({ ...point, width: 0, height: 0 })
                  setSelectedId(null)
                }}
                onUpdateDrawing={(point) => {
                  if (drawStartRef.current)
                    setDraftRect(rectFromPoints(drawStartRef.current, point))
                }}
                onFinishDrawing={finishDrawing}
                onSelectAnnotation={setSelectedId}
                onUpdateAnnotationRect={(id, rect) => {
                  if (!busy) updateBox(id, rect)
                }}
                onAnnotationInteractionStart={beginAnnotationInteraction}
                onAnnotationInteractionEnd={endAnnotationInteraction}
                onChangeAnnotationLabel={(id, next) => {
                  if (!busy)
                    updateBox(id, {
                      label: next,
                      color: labelToColor(next),
                      sourceClassIndex: null,
                      hasUnknownClass: false,
                    })
                }}
                onDuplicateAnnotation={(id) => {
                  if (!busy) duplicateBox(id)
                }}
                onDeleteAnnotation={(id) => {
                  if (!busy) deleteBox(id)
                }}
                onHoverPointChange={(point) => {
                  hoverPointRef.current = point
                }}
              />
            ) : null}
            {!source ? (
              <div className="video-empty">
                <svg className="video-empty-icon" width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4m-7-6 5 3-5 3Z" /></svg>
                <h3>Open a video</h3>
                <p>
                  Drop a file here or select one to annotate its frames.
                </p>
                <AppButton
                  variant="primary"
                  disabled={busy}
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose file
                </AppButton>
                <p className="muted">
                  Saved frames will be added to {datasetName}.
                </p>
              </div>
            ) : null}
            {busy || seeking ? (
              <div className="video-stage-busy" role="status">
                {busy ? 'Processing frame…' : 'Seeking…'}
              </div>
            ) : null}
          </div>
          <aside className="video-inspector">
            <label>
              Active class
              <input
                list="video-class-options"
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                disabled={busy}
              />
            </label>
            <datalist id="video-class-options">
              {classOptions.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
            {selected ? (
              <div className="video-selection">
                <strong>Selected box</strong>
                <label>
                  Class
                  <input
                    list="video-class-options"
                    value={selected.label}
                    onFocus={() => annotationHistory.beginGroup('frame', false)}
                    onBlur={endAnnotationInteraction}
                    disabled={busy}
                    onChange={(event) =>
                      updateBox(selected.id, {
                        label: event.target.value,
                        color: labelToColor(event.target.value),
                        sourceClassIndex: null,
                        hasUnknownClass: false,
                      })
                    }
                  />
                </label>
                <label className="video-checkbox">
                  <input
                    type="checkbox"
                    checked={selected.difficult}
                    disabled={busy}
                    onChange={(event) =>
                      updateBox(selected.id, {
                        difficult: event.target.checked,
                      })
                    }
                  />
                  Difficult
                </label>
                <div className="video-actions">
                  <AppButton
                    disabled={busy}
                    onClick={() => duplicateBox(selected.id)}
                  >
                    Duplicate
                  </AppButton>
                  <AppButton
                    disabled={busy}
                    onClick={() => deleteBox(selected.id)}
                  >
                    Delete
                  </AppButton>
                </div>
              </div>
            ) : null}
            <div className="video-boxes" aria-label="Frame annotations">
              <strong>{annotations.length} boxes</strong>
              {annotations.map((box, index) => (
                <button
                  key={box.id}
                  className={box.id === selectedId ? 'is-selected' : ''}
                  onClick={() => setSelectedId(box.id)}
                >
                  {index + 1}. {box.label}
                  {box.difficult ? ' · difficult' : ''}
                </button>
              ))}
            </div>
            <AppButton
              disabled={
                !frame || busy || !lastBoxes.length || !!annotations.length
              }
              onClick={() => {
                if (!frame) return
                edit(
                  lastBoxes.map((box) => ({
                    ...box,
                    id: crypto.randomUUID(),
                    x: Math.min(box.x, frame.image.width - 1),
                    y: Math.min(box.y, frame.image.height - 1),
                    width: Math.min(
                      box.width,
                      frame.image.width -
                        Math.min(box.x, frame.image.width - 1),
                    ),
                    height: Math.min(
                      box.height,
                      frame.image.height -
                        Math.min(box.y, frame.image.height - 1),
                    ),
                  })),
                )
              }}
            >
              Copy last saved boxes
            </AppButton>
            <label>
              Image format
              <select
                value={format}
                disabled={busy}
                onChange={(event) =>
                  setFormat(event.target.value as 'png' | 'jpg')
                }
              >
                <option value="png">PNG · lossless</option>
                <option value="jpg">JPEG · quality 95%</option>
              </select>
            </label>
            <p className="muted">
              Format applies to the next frame. Frames keep their original
              resolution.
            </p>
            <AppButton
              variant="primary"
              disabled={
                !frame || busy || seeking || annotations.length === 0 || (!!frame.savedImage && !dirty)
              }
              onClick={() => void save()}
            >
              {frame?.savedImage ? 'Save changes' : 'Save frame to dataset'}
            </AppButton>
            <AppButton
              disabled={!frame || busy || seeking || annotations.length === 0}
              onClick={async () => {
                if ((frame?.savedImage && !dirty) || (await save()))
                  seek(time + sampleInterval)
              }}
            >
              Save & next interval
            </AppButton>
            <AppButton
              disabled={!frame || busy || seeking || annotations.length > 0 || (!!frame.savedImage && !dirty)}
              onClick={() => void save('background')}
            >
              Save as background
            </AppButton>
            <p className="video-save-status" role="status">
              {message}
            </p>
            <p className="muted">
              Empty frames are saved only with Save as background. Playback and seeking do not add frames to the dataset.
            </p>
          </aside>
        </div>
        <footer className="video-transport" inert={confirmNavigation}>
          <div className="video-timeline-row">
            <output>{formatVideoTime(time)}</output>
            <input
              aria-label="Video timeline"
              type="range"
              min="0"
              max={duration || 1}
              step="0.001"
              value={Math.min(timelineDraft ?? time, duration || 1)}
              disabled={!duration || busy || seeking || confirmNavigation}
              onChange={(event) => {
                setTimelineDraft(Number(event.target.value))
              }}
              onPointerUp={(event) => commitTimeline(event.currentTarget.value)}
              onPointerCancel={() => setTimelineDraft(null)}
              onKeyUp={(event) => {
                if (
                  [
                    'ArrowLeft',
                    'ArrowRight',
                    'ArrowUp',
                    'ArrowDown',
                    'Home',
                    'End',
                    'PageUp',
                    'PageDown',
                  ].includes(event.key)
                )
                  commitTimeline(event.currentTarget.value)
              }}
            />
            <output>{formatVideoTime(duration)}</output>
          </div>
          <div className="video-transport-controls">
            <AppButton
              disabled={!duration || busy || seeking}
              onClick={togglePlayback}
            >
              {playing ? 'Pause & annotate' : 'Play'}
            </AppButton>
            <AppButton
              title="Left arrow"
              disabled={!duration || busy || seeking}
              onClick={() => navigate(() => seek(time - 1 / frameRate))}
            >
              ← Step
            </AppButton>
            <AppButton
              title="Right arrow"
              disabled={!duration || busy || seeking}
              onClick={() => navigate(() => seek(time + 1 / frameRate))}
            >
              Step →
            </AppButton>
            <label title="Set the source FPS for time-based frame stepping">
              Step FPS
              <input
                type="number"
                min="1"
                max="240"
                value={fps}
                onChange={(event) => setFps(event.target.value)}
              />
            </label>
            <label>
              Speed
              <select
                value={speed}
                onChange={(event) => {
                  setSpeed(event.target.value)
                  if (videoRef.current)
                    videoRef.current.playbackRate = Number(event.target.value)
                }}
              >
                {['0.25', '0.5', '1', '1.5', '2', '4'].map((value) => (
                  <option key={value} value={value}>
                    {value}×
                  </option>
                ))}
              </select>
            </label>
            <label>
              Interval (s)
              <input
                type="number"
                min="0.001"
                step="0.1"
                value={interval}
                onChange={(event) => setInterval(event.target.value)}
              />
            </label>
            <AppButton
              title="Shift + left arrow"
              disabled={!duration || busy || seeking}
              onClick={() => navigate(() => seek(time - sampleInterval))}
            >
              − Interval
            </AppButton>
            <AppButton
              title="Shift + right arrow"
              disabled={!duration || busy || seeking}
              onClick={() => navigate(() => seek(time + sampleInterval))}
            >
              + Interval
            </AppButton>
            <label>
              Go to (s)
              <input
                type="number"
                min="0"
                max={duration}
                step="0.001"
                value={jumpTime}
                onChange={(event) => setJumpTime(event.target.value)}
              />
            </label>
            <AppButton
              disabled={
                !duration ||
                busy ||
                seeking ||
                !Number.isFinite(Number(jumpTime))
              }
              onClick={() => navigate(() => seek(Number(jumpTime)))}
            >
              Go
            </AppButton>
          </div>
          <div className="video-saved-strip">
            <strong>{Object.keys(savedFrames).length} saved</strong>
            {Object.entries(savedFrames)
              .sort((a, b) => a[1].time - b[1].time)
              .map(([key, saved]) => (
                <button
                  key={key}
                  disabled={busy || seeking}
                  title={saved.image.name}
                  onClick={() => navigate(() => seek(saved.time + 0.000001))}
                >
                  {formatVideoTime(saved.time)}
                </button>
              ))}
          </div>
          <p className="video-shortcuts">
            Space: play / pause · ← →: time step · Shift + ← →: interval ·
            History: {undoShortcut?.split(' / ')[0] || 'Undo'} / {redoShortcut?.split(' / ')[0] || 'Redo'} · Ctrl+S: save · 1–9: class · W: new box · Delete: box · Esc: cancel /
            close. Set Step FPS to match your video.
          </p>
        </footer>
        {confirmNavigation ? (
          <div
            className="video-confirm"
            role="alertdialog"
            aria-modal="true"
            aria-label="Unsaved frame annotations"
          >
            <div>
              <h3>Save this frame before continuing?</h3>
              <p>
                Your current annotations have not been saved to the dataset.
              </p>
              <div className="video-actions">
                <AppButton
                  disabled={busy}
                  onClick={() => {
                    setConfirmNavigation(false)
                    pendingActionRef.current = null
                  }}
                >
                  Keep editing
                </AppButton>
                <AppButton
                  disabled={busy}
                  onClick={() => void continueNavigation(false)}
                >
                  Discard changes
                </AppButton>
                <AppButton
                  variant="primary"
                  disabled={busy}
                  onClick={() => void continueNavigation(true, annotations.length === 0 ? 'background' : 'annotations')}
                >
                  {annotations.length === 0 ? 'Save as background & continue' : 'Save & continue'}
                </AppButton>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
