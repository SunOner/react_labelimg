import {
  memo,
  useCallback,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { AppButton } from './ui/AppButton'
import { MIN_ANNOTATION_SIZE } from '../lib/annotations'
import type { Annotation, LoadedImage, Point, Rect } from '../types'

const STAGE_WIDTH = 1600
const STAGE_HEIGHT = 900
const MIN_ZOOM = 1
const MAX_ZOOM = 8
const ZOOM_STEP = 0.25
const BOX_DRAG_ARM_THRESHOLD_PX = 4
const CONTEXT_SUBMENU_CLOSE_DELAY_MS = 180

type AnnotationViewportProps = {
  image: LoadedImage | null
  imageLabel: string | null
  isLoading: boolean
  isError?: boolean
  annotations: Annotation[]
  selectedId: string | null
  draftRect: Rect | null
  tool: 'draw' | 'new-box' | 'sam-click' | 'sam-box'
  showSamTools: boolean
  isSamBusy: boolean
  classOptions: string[]
  onSelectTool: (tool: 'draw' | 'new-box' | 'sam-click' | 'sam-box') => void
  onOpenDataset: () => void
  recentDatasets: Array<{ path: string; label: string }>
  onOpenRecentDataset: (path: string) => void
  onRemoveRecentDataset: (path: string) => void
  openDatasetDisabled?: boolean
  onStartDrawing: (point: Point) => void
  onUpdateDrawing: (point: Point) => void
  onFinishDrawing: (point: Point) => void
  onSelectAnnotation: (annotationId: string) => void
  onUpdateAnnotationRect: (annotationId: string, rect: Rect) => void
  onChangeAnnotationLabel: (annotationId: string, nextLabel: string) => void
  onDuplicateAnnotation: (annotationId: string) => void
  onDeleteAnnotation: (annotationId: string) => void
  onHoverPointChange?: (point: Point | null) => void
}

type ResizeHandle = 'nw' | 'ne' | 'sw' | 'se'

type AnnotationLabelMetrics = {
  fontSize: number
  textStrokeWidth: number
  characterWidth: number
  tagHeight: number
  tagPaddingX: number
  tagRadius: number
  labelGap: number
  minTagWidth: number
}

type AnnotationGeometry = {
  annotation: Annotation
  stageRect: Rect
  labelRect: Rect
  labelMetrics: AnnotationLabelMetrics
}

type AnnotationInteraction =
  | {
      kind: 'pan'
      startClient: Point
      originPan: Point
    }
  | {
      kind: 'pending-select'
      annotationId: string
      startClient: Point
      startPoint: Point
    }
  | {
      kind: 'move'
      annotationId: string
      startPoint: Point
      originRect: Rect
    }
  | {
      kind: 'resize'
      annotationId: string
      handle: ResizeHandle
      startPoint: Point
      originRect: Rect
    }

export const AnnotationViewport = memo(function AnnotationViewport({
  image,
  imageLabel,
  isLoading,
  isError = false,
  annotations,
  selectedId,
  draftRect,
  tool,
  showSamTools,
  isSamBusy,
  classOptions,
  onSelectTool,
  onOpenDataset,
  recentDatasets,
  onOpenRecentDataset,
  onRemoveRecentDataset,
  openDatasetDisabled = false,
  onStartDrawing,
  onUpdateDrawing,
  onFinishDrawing,
  onSelectAnnotation,
  onUpdateAnnotationRect,
  onChangeAnnotationLabel,
  onDuplicateAnnotation,
  onDeleteAnnotation,
  onHoverPointChange,
}: AnnotationViewportProps) {
  const resizeObserverRef = useRef<ResizeObserver | null>(null)
  const stageViewportRef = useRef<HTMLDivElement | null>(null)
  const overlayRef = useRef<SVGSVGElement | null>(null)
  const contextMenuRef = useRef<HTMLDivElement | null>(null)
  const contextSubmenuCloseTimeoutRef = useRef<number | null>(null)
  const interactionRef = useRef<AnnotationInteraction | null>(null)
  const selectionCycleRef = useRef<{
    point: Point
    candidateIds: string[]
  } | null>(null)
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [isPanning, setIsPanning] = useState(false)
  const [contextMenu, setContextMenu] = useState<{
    annotationId: string
    x: number
    y: number
  } | null>(null)
  const [isContextSubmenuOpen, setIsContextSubmenuOpen] = useState(false)
  const autoFitZoom = image
    ? getAutoFitZoom(image.width, image.height)
    : MIN_ZOOM

  const setStageViewportNode = useCallback((node: HTMLDivElement | null) => {
    resizeObserverRef.current?.disconnect()
    resizeObserverRef.current = null
    stageViewportRef.current = node

    if (!node) {
      return
    }

    const syncViewportSize = () => {
      const width = node.clientWidth
      const height = node.clientHeight
      setViewportSize((current) =>
        current.width === width && current.height === height
          ? current
          : { width, height },
      )
    }

    syncViewportSize()

    if (typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(() => {
      syncViewportSize()
    })
    observer.observe(node)
    resizeObserverRef.current = observer
  }, [])

  useEffect(() => {
    return () => {
      resizeObserverRef.current?.disconnect()
      if (contextSubmenuCloseTimeoutRef.current !== null) {
        window.clearTimeout(contextSubmenuCloseTimeoutRef.current)
      }
    }
  }, [])

  const clearContextSubmenuCloseTimeout = () => {
    if (contextSubmenuCloseTimeoutRef.current !== null) {
      window.clearTimeout(contextSubmenuCloseTimeoutRef.current)
      contextSubmenuCloseTimeoutRef.current = null
    }
  }

  const openContextSubmenu = () => {
    clearContextSubmenuCloseTimeout()
    setIsContextSubmenuOpen(true)
  }

  const scheduleContextSubmenuClose = () => {
    clearContextSubmenuCloseTimeout()
    contextSubmenuCloseTimeoutRef.current = window.setTimeout(() => {
      setIsContextSubmenuOpen(false)
      contextSubmenuCloseTimeoutRef.current = null
    }, CONTEXT_SUBMENU_CLOSE_DELAY_MS)
  }

  useEffect(() => {
    clearContextSubmenuCloseTimeout()
    setIsContextSubmenuOpen(false)
  }, [contextMenu])

  useEffect(() => {
    if (!image) {
      return
    }

    interactionRef.current = null
    setIsPanning(false)
    setContextMenu(null)
    setPan({ x: 0, y: 0 })
    setZoom(autoFitZoom)
  }, [autoFitZoom, image?.height, image?.id, image?.width])

  const onWindowPointerMove = useEffectEvent((event: PointerEvent) => {
    const interaction = interactionRef.current
    if (!interaction) {
      return
    }

    if (interaction.kind === 'pan') {
      setPan(
        clampPan(
          {
            x: interaction.originPan.x + event.clientX - interaction.startClient.x,
            y: interaction.originPan.y + event.clientY - interaction.startClient.y,
          },
          viewportSize,
          zoom,
        ),
      )
      return
    }

    if (interaction.kind === 'pending-select') {
      return
    }

    if (!image) {
      return
    }

    const overlay = overlayRef.current
    if (!overlay) {
      return
    }

    const point = pointFromClient(
      event.clientX,
      event.clientY,
      overlay.getBoundingClientRect(),
      image,
      fitContainRect(image.width, image.height, STAGE_WIDTH, STAGE_HEIGHT),
      'clamp',
    )

    if (!point) {
      return
    }

    if (interaction.kind === 'move') {
      onUpdateAnnotationRect(
        interaction.annotationId,
        clampRectToImage(
          {
            x: interaction.originRect.x + point.x - interaction.startPoint.x,
            y: interaction.originRect.y + point.y - interaction.startPoint.y,
            width: interaction.originRect.width,
            height: interaction.originRect.height,
          },
          image,
        ),
      )
      return
    }

    onUpdateAnnotationRect(
      interaction.annotationId,
      resizeRectFromHandle(interaction.originRect, interaction.handle, point, interaction.startPoint, image),
    )
  })

  const onWindowPointerUp = useEffectEvent(() => {
    interactionRef.current = null
    setIsPanning(false)
  })

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      onWindowPointerMove(event)
    }

    const handlePointerUp = () => {
      onWindowPointerUp()
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    window.addEventListener('pointercancel', handlePointerUp)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
      window.removeEventListener('pointercancel', handlePointerUp)
    }
  }, [])

  useEffect(() => {
    if (!contextMenu) {
      return
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (
        contextMenuRef.current &&
        event.target instanceof Node &&
        !contextMenuRef.current.contains(event.target)
      ) {
        setContextMenu(null)
      }
    }

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setContextMenu(null)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleEscape)
    }
  }, [contextMenu])

  const canZoomOut = zoom > MIN_ZOOM
  const canZoomIn = zoom < MAX_ZOOM
  const zoomLabel = `${Math.round(zoom * 100)}%`
  const isDrawTool = tool === 'draw' || tool === 'new-box'
  const isEditTool = tool === 'draw'
  const canUseSamBox = tool === 'sam-box' && !isSamBusy
  const overlayClassName = [
    'viewport-overlay',
    isPanning ? 'is-panning' : '',
    tool === 'sam-click' ? 'is-sam-click' : '',
    tool === 'sam-box' ? 'is-sam-box' : '',
  ]
    .filter(Boolean)
    .join(' ')

  const updateZoom = (
    nextZoom: number,
    focalPoint?: { clientX: number; clientY: number },
  ) => {
    const clampedZoom = clampZoom(nextZoom)
    if (clampedZoom === zoom) {
      return
    }

    const viewport = stageViewportRef.current
    if (!viewport) {
      setZoom(clampedZoom)
      return
    }

    const bounds = viewport.getBoundingClientRect()
    const liveBounds = {
      width: viewport.clientWidth || 1,
      height: viewport.clientHeight || 1,
    }
    const currentPan = clampPan(pan, liveBounds, zoom)
    const currentOffset = getStageOffset(liveBounds, zoom)
    const pointerX = focalPoint ? focalPoint.clientX - bounds.left : bounds.width / 2
    const pointerY = focalPoint ? focalPoint.clientY - bounds.top : bounds.height / 2
    const stageX = (pointerX - currentOffset.x - currentPan.x) / zoom
    const stageY = (pointerY - currentOffset.y - currentPan.y) / zoom
    const nextOffset = getStageOffset(liveBounds, clampedZoom)
    const nextPan = clampPan(
      {
        x: pointerX - nextOffset.x - stageX * clampedZoom,
        y: pointerY - nextOffset.y - stageY * clampedZoom,
      },
      liveBounds,
      clampedZoom,
    )

    setZoom(clampedZoom)
    setPan(nextPan)
  }

  const handleViewportWheel = useEffectEvent((event: WheelEvent) => {
    event.preventDefault()
    const nextZoom = zoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP)
    updateZoom(nextZoom, {
      clientX: event.clientX,
      clientY: event.clientY,
    })
  })

  useEffect(() => {
    const viewport = stageViewportRef.current
    if (!viewport) {
      return
    }

    const handleWheel = (event: WheelEvent) => {
      handleViewportWheel(event)
    }

    viewport.addEventListener('wheel', handleWheel, { passive: false })
    return () => {
      viewport.removeEventListener('wheel', handleWheel)
    }
  }, [handleViewportWheel, viewportSize.width, viewportSize.height])

  if (!image) {
    const emptyStateLabel = isLoading
      ? `Loading ${imageLabel ?? 'image'}`
      : isError
        ? `Failed to load ${imageLabel ?? 'image'}`
        : 'No image loaded'

    return (
      <section
        className="viewport viewport-empty"
        aria-label={emptyStateLabel}
      >
        <div className="empty-state">
          {recentDatasets.length > 0 ? (
            <div className="recent-datasets" aria-label="Recent datasets">
              <div className="recent-datasets-title">Recent datasets</div>
              {recentDatasets.map((dataset) => (
                <div key={dataset.path} className="recent-dataset-row">
                  <button
                    type="button"
                    className="recent-dataset-button"
                    onClick={() => onOpenRecentDataset(dataset.path)}
                    disabled={openDatasetDisabled}
                    title={dataset.path}
                  >
                    <span className="recent-dataset-label">{dataset.label}</span>
                    <span className="recent-dataset-path">{dataset.path}</span>
                  </button>
                  <button
                    type="button"
                    className="recent-dataset-remove"
                    onClick={() => onRemoveRecentDataset(dataset.path)}
                    aria-label={`Remove ${dataset.label} from recent datasets`}
                    title="Remove from history"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          {!isLoading ? (
            <div className="empty-state-action-wrap">
              <AppButton
                variant="primary"
                className="empty-state-action"
                onClick={onOpenDataset}
                disabled={openDatasetDisabled}
              >
                Open dataset
              </AppButton>
            </div>
          ) : null}
        </div>
        <span className="visually-hidden">{emptyStateLabel}</span>
      </section>
    )
  }

  const stageImagePlacement =
    fitContainRect(image.width, image.height, STAGE_WIDTH, STAGE_HEIGHT)
  const liveViewport = {
    width: viewportSize.width || 1,
    height: viewportSize.height || 1,
  }
  const overlayMetrics = getOverlayMetrics(liveViewport, zoom)
  const stageOffset = getStageOffset(liveViewport, zoom)
  const resolvedPan = clampPan(pan, liveViewport, zoom)
  const shellWidth = snapCssPixel(liveViewport.width * zoom)
  const shellHeight = snapCssPixel(liveViewport.height * zoom)
  const shellTranslateX = snapCssPixel(stageOffset.x + resolvedPan.x)
  const shellTranslateY = snapCssPixel(stageOffset.y + resolvedPan.y)
  const borderHitTolerance = getBorderHitTolerance(shellWidth)
  const annotationGeometries = layoutAnnotationLabels(
    annotations.map((annotation) =>
      buildAnnotationGeometry(annotation, image, stageImagePlacement, overlayMetrics),
    ),
    overlayMetrics,
  )
  const contextMenuPosition = contextMenu
    ? {
        left: Math.max(6, Math.min(contextMenu.x, liveViewport.width - 150)),
        top: Math.max(6, Math.min(contextMenu.y, liveViewport.height - 108)),
      }
    : null
  const contextMenuAnnotation = contextMenu
    ? annotations.find((annotation) => annotation.id === contextMenu.annotationId) ?? null
    : null
  const contextMenuClassOptions = contextMenuAnnotation
    ? classOptions.filter((label) => label !== contextMenuAnnotation.label)
    : []
  const contextMenuSubmenuClassName = [
    'annotation-context-submenu',
    isContextSubmenuOpen ? 'is-open' : '',
    contextMenu && contextMenu.x > liveViewport.width - 296 ? 'is-left' : '',
  ]
    .filter(Boolean)
    .join(' ')
  const isAtAutoFitZoom =
    Math.abs(zoom - autoFitZoom) < 0.001 &&
    Math.abs(resolvedPan.x) < 0.5 &&
    Math.abs(resolvedPan.y) < 0.5

  const handleZoomOut = () => updateZoom(zoom - ZOOM_STEP)

  const handleZoomIn = () => updateZoom(zoom + ZOOM_STEP)

  const handleZoomReset = () => {
    setPan({ x: 0, y: 0 })
    setZoom(autoFitZoom)
  }

  const handleOverlayPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    const stagePoint = stagePointFromEvent(event)
    const point = pointFromEvent(event, image, stageImagePlacement, 'strict')
    const clampedPoint = pointFromEvent(event, image, stageImagePlacement, 'clamp')
    const hitCandidates =
      stagePoint
        ? getAnnotationCandidatesAtStagePoint(
            annotationGeometries,
            stagePoint,
            borderHitTolerance,
          )
        : []
    const hitAnnotation =
      stagePoint
        ? resolveAnnotationAtPoint(
            hitCandidates,
            stagePoint,
            selectedId,
            selectionCycleRef.current,
          )
        : null

    if (event.button === 2 && zoom > MIN_ZOOM) {
      if (hitAnnotation) {
        return
      }

      event.preventDefault()
      setContextMenu(null)
      event.currentTarget.setPointerCapture(event.pointerId)
      interactionRef.current = {
        kind: 'pan',
        startClient: { x: event.clientX, y: event.clientY },
        originPan: resolvedPan,
      }
      setIsPanning(true)
      return
    }

    if (event.button !== 0) {
      return
    }

    event.preventDefault()
    setContextMenu(null)

    if (!isDrawTool) {
      selectionCycleRef.current = null

      if (tool === 'sam-click') {
        if (!point) {
          return
        }

        onStartDrawing(point)
        return
      }

      if (canUseSamBox) {
        if (!clampedPoint) {
          return
        }

        event.currentTarget.setPointerCapture(event.pointerId)
        onStartDrawing(point ?? clampedPoint)
      }
      return
    }

    if (tool === 'new-box') {
      if (!clampedPoint) {
        selectionCycleRef.current = null
        return
      }

      selectionCycleRef.current = null
      onFinishDrawing(point ?? clampedPoint)
      return
    }

    if (hitAnnotation) {
      if (!stagePoint) {
        return
      }

      selectionCycleRef.current = {
        point: stagePoint,
        candidateIds: hitCandidates.map((candidate) => candidate.id),
      }

      if (!point) {
        onSelectAnnotation(hitAnnotation.id)
        return
      }

      if (hitAnnotation.id === selectedId) {
        onSelectAnnotation(hitAnnotation.id)
        interactionRef.current = {
          kind: 'move',
          annotationId: hitAnnotation.id,
          startPoint: point,
          originRect: {
            x: hitAnnotation.x,
            y: hitAnnotation.y,
            width: hitAnnotation.width,
            height: hitAnnotation.height,
          },
        }
        return
      }

      event.currentTarget.setPointerCapture(event.pointerId)
      interactionRef.current = {
        kind: 'pending-select',
        annotationId: hitAnnotation.id,
        startClient: { x: event.clientX, y: event.clientY },
        startPoint: point,
      }
      return
    }

    if (!clampedPoint) {
      selectionCycleRef.current = null
      return
    }

    selectionCycleRef.current = null
    event.currentTarget.setPointerCapture(event.pointerId)
    onStartDrawing(point ?? clampedPoint)
  }

  const handleOverlayPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const hoverPoint = pointFromEvent(event, image, stageImagePlacement, 'strict')
    const clampedPoint = pointFromEvent(event, image, stageImagePlacement, 'clamp')
    onHoverPointChange?.(hoverPoint)

    const interaction = interactionRef.current

    if (interaction?.kind === 'pan') {
      return
    }

    if (interaction?.kind === 'pending-select') {
      if (event.buttons !== 1) {
        return
      }

      const deltaX = Math.abs(event.clientX - interaction.startClient.x)
      const deltaY = Math.abs(event.clientY - interaction.startClient.y)
      if (
        deltaX < BOX_DRAG_ARM_THRESHOLD_PX &&
        deltaY < BOX_DRAG_ARM_THRESHOLD_PX
      ) {
        return
      }

      if (!clampedPoint) {
        return
      }

      selectionCycleRef.current = null
      interactionRef.current = null
      onStartDrawing(interaction.startPoint)
      onUpdateDrawing(clampedPoint)
      return
    }

    if (tool === 'sam-click') {
      return
    }

    if (tool === 'new-box') {
      if (!clampedPoint) {
        return
      }

      onUpdateDrawing(clampedPoint)
      return
    }

    handlePointerMove(event, image, stageImagePlacement, onUpdateDrawing)
  }

  const handleOverlayPointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    const interaction = interactionRef.current

    if (interaction?.kind === 'pan') {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
      interactionRef.current = null
      setIsPanning(false)
      return
    }

    if (interaction?.kind === 'pending-select') {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }

      interactionRef.current = null
      if (event.type !== 'pointercancel') {
        onSelectAnnotation(interaction.annotationId)
      }
      return
    }

    if (tool === 'sam-click') {
      return
    }

    if (tool === 'new-box') {
      return
    }

    handlePointerUp(event, image, stageImagePlacement, onFinishDrawing)
  }

  const handleOverlayPointerLeave = () => {
    onHoverPointChange?.(null)
  }

  const handleOverlayContextMenu = (event: ReactPointerEvent<SVGSVGElement>) => {
    event.preventDefault()

    const point = stagePointFromEvent(event)
    const hitCandidates =
      point
        ? getAnnotationCandidatesAtStagePoint(
            annotationGeometries,
            point,
            borderHitTolerance,
          )
        : []
    const hitAnnotation = hitCandidates[0] ?? null

    if (!hitAnnotation) {
      setContextMenu(null)
      selectionCycleRef.current = null
      return
    }

    if (point) {
      selectionCycleRef.current = {
        point,
        candidateIds: hitCandidates.map((candidate) => candidate.id),
      }
    }

    const bounds = stageViewportRef.current?.getBoundingClientRect()
    if (!bounds) {
      return
    }

    onSelectAnnotation(hitAnnotation.id)
    setContextMenu({
      annotationId: hitAnnotation.id,
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    })
  }

  return (
    <section className="viewport">
      <div className="viewport-toolbar" aria-label="Canvas zoom controls">
        <div className="viewport-zoom-group">
          <AppButton
            variant="menu-trigger"
            onClick={handleZoomOut}
            disabled={!canZoomOut}
            aria-label="Zoom out"
            title="Zoom out"
          >
            -
          </AppButton>
          <span className="viewport-zoom-value">{zoomLabel}</span>
          <AppButton
            variant="menu-trigger"
            onClick={handleZoomReset}
            disabled={isAtAutoFitZoom}
            aria-label={`Fit image (${Math.round(autoFitZoom * 100)}%)`}
            title={`Fit image (${Math.round(autoFitZoom * 100)}%)`}
          >
            Fit
          </AppButton>
          <AppButton
            variant="menu-trigger"
            onClick={handleZoomIn}
            disabled={!canZoomIn}
            aria-label="Zoom in"
            title="Zoom in"
          >
            +
          </AppButton>
        </div>
      </div>

      <div className="viewport-body">
        <div className="viewport-tools" aria-label="Viewport tools">
          <button
            type="button"
            className={isDrawTool ? 'viewport-tool is-active' : 'viewport-tool'}
            onClick={() => onSelectTool('draw')}
            title="Draw new box (W)"
            aria-label="Draw new box (W)"
          >
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <rect x="4" y="4" width="12" height="12" rx="1.5" />
              <path d="M10 2.5v3M10 14.5v3M2.5 10h3M14.5 10h3" />
            </svg>
          </button>
          {showSamTools ? (
            <>
              <button
                type="button"
                className={
                  tool === 'sam-click' ? 'viewport-tool is-active' : 'viewport-tool'
                }
                onClick={() => onSelectTool('sam-click')}
                title="SAM click select"
                aria-label="SAM click select"
                disabled={isSamBusy}
              >
                <svg viewBox="0 0 20 20" aria-hidden="true">
                  <circle cx="10" cy="10" r="4.25" />
                  <path d="M10 2.5v3M10 14.5v3M2.5 10h3M14.5 10h3" />
                </svg>
              </button>
              <button
                type="button"
                className={
                  tool === 'sam-box' ? 'viewport-tool is-active' : 'viewport-tool'
                }
                onClick={() => onSelectTool('sam-box')}
                title="SAM box select"
                aria-label="SAM box select"
                disabled={isSamBusy}
              >
                <svg viewBox="0 0 20 20" aria-hidden="true">
                  <rect x="4.25" y="4.25" width="11.5" height="11.5" rx="1.5" />
                  <path d="M2.5 6.5h3M14.5 13.5h3M6.5 2.5v3M13.5 14.5v3" />
                </svg>
              </button>
            </>
          ) : null}
        </div>

        <div
          ref={setStageViewportNode}
          className="viewport-scroll"
          onContextMenu={(event) => event.preventDefault()}
        >
          <div
            className="viewport-stage-shell"
            style={{
              width: `${shellWidth}px`,
              height: `${shellHeight}px`,
              transform: `translate3d(${shellTranslateX}px, ${shellTranslateY}px, 0)`,
            }}
          >
            <svg
              className="viewport-stage"
              viewBox={`0 0 ${STAGE_WIDTH} ${STAGE_HEIGHT}`}
              preserveAspectRatio="xMidYMid meet"
            >
              <rect
                className="viewport-image-frame"
                x={stageImagePlacement.x}
                y={stageImagePlacement.y}
                width={stageImagePlacement.width}
                height={stageImagePlacement.height}
              />
              <image
                className="viewport-image"
                href={image.url}
                x={stageImagePlacement.x}
                y={stageImagePlacement.y}
                width={stageImagePlacement.width}
                height={stageImagePlacement.height}
                preserveAspectRatio="none"
              />
            </svg>

            <svg
              ref={overlayRef}
              className={overlayClassName}
              viewBox={`0 0 ${STAGE_WIDTH} ${STAGE_HEIGHT}`}
              preserveAspectRatio="xMidYMid meet"
              pointerEvents="all"
              shapeRendering="geometricPrecision"
              textRendering="geometricPrecision"
              onPointerDown={handleOverlayPointerDown}
              onPointerMove={handleOverlayPointerMove}
              onPointerUp={handleOverlayPointerUp}
              onPointerCancel={handleOverlayPointerUp}
              onPointerLeave={handleOverlayPointerLeave}
              onContextMenu={handleOverlayContextMenu}
            >
              {annotationGeometries.map((geometry) => {
              const { annotation, stageRect, labelRect, labelMetrics } = geometry
              const isSelected = annotation.id === selectedId
              const connector = getLabelConnector(labelRect, stageRect)

              return (
                <g key={annotation.id} pointerEvents="none">
                  <rect
                    className={
                      isSelected ? 'bbox-rect is-selected' : 'bbox-rect'
                    }
                    x={stageRect.x}
                    y={stageRect.y}
                    width={stageRect.width}
                    height={stageRect.height}
                    stroke={annotation.color}
                    fill={withOpacity(annotation.color, isSelected ? 0.26 : 0.14)}
                    pointerEvents="none"
                  />
                  <line
                    className="bbox-tag-connector"
                    x1={connector.start.x}
                    y1={connector.start.y}
                    x2={connector.end.x}
                    y2={connector.end.y}
                    stroke={annotation.color}
                    pointerEvents="none"
                  />
                  <g
                    transform={`translate(${labelRect.x}, ${labelRect.y})`}
                    pointerEvents="none"
                  >
                    <rect
                      className="bbox-tag"
                      width={labelRect.width}
                      height={labelRect.height}
                      rx={labelMetrics.tagRadius}
                      ry={labelMetrics.tagRadius}
                      fill={annotation.color}
                    />
                    <text
                      className="bbox-text"
                      x={labelMetrics.tagPaddingX}
                      y={labelRect.height / 2}
                      fontSize={labelMetrics.fontSize}
                      strokeWidth={labelMetrics.textStrokeWidth}
                      dominantBaseline="middle"
                      paintOrder="stroke fill"
                    >
                      {annotation.label || 'object'}
                    </text>
                  </g>
                  {isEditTool && isSelected
                    ? buildResizeHandles(stageRect, overlayMetrics.effectiveScale).map((handle) => (
                        <g key={handle.id}>
                          <circle
                            className={`bbox-handle-hit bbox-handle-${handle.id}`}
                            cx={handle.cx}
                            cy={handle.cy}
                            r={handle.hitRadius}
                            pointerEvents="all"
                            onPointerDown={(event) => {
                              if (event.button !== 0) {
                                return
                              }

                              const point = pointFromEvent(
                                event,
                                image,
                                stageImagePlacement,
                                'clamp',
                              )
                              if (!point) {
                                return
                              }

                              event.preventDefault()
                              event.stopPropagation()
                              setContextMenu(null)
                              onSelectAnnotation(annotation.id)
                              interactionRef.current = {
                                kind: 'resize',
                                annotationId: annotation.id,
                                handle: handle.id,
                                startPoint: point,
                                originRect: {
                                  x: annotation.x,
                                  y: annotation.y,
                                  width: annotation.width,
                                  height: annotation.height,
                                },
                              }
                            }}
                            onContextMenu={(event) => {
                              event.preventDefault()
                              event.stopPropagation()
                              const bounds =
                                stageViewportRef.current?.getBoundingClientRect()
                              if (!bounds) {
                                return
                              }

                              setContextMenu({
                                annotationId: annotation.id,
                                x: event.clientX - bounds.left,
                                y: event.clientY - bounds.top,
                              })
                            }}
                          />
                          <circle
                            className={`bbox-handle bbox-handle-${handle.id}`}
                            cx={handle.cx}
                            cy={handle.cy}
                            r={handle.radius}
                            strokeWidth={handle.strokeWidth}
                            pointerEvents="none"
                          />
                        </g>
                      ))
                    : null}
                </g>
              )
            })}

              {draftRect ? (
                <rect
                  className="bbox-rect is-draft"
                  x={stageImagePlacement.x + draftRect.x * stageImagePlacement.scale}
                  y={stageImagePlacement.y + draftRect.y * stageImagePlacement.scale}
                  width={draftRect.width * stageImagePlacement.scale}
                  height={draftRect.height * stageImagePlacement.scale}
                  stroke="var(--accent)"
                  fill="rgba(54, 95, 72, 0.12)"
                  pointerEvents="none"
                />
              ) : null}
            </svg>
          </div>
          {contextMenu && contextMenuPosition ? (
            <div
              ref={contextMenuRef}
              className="annotation-context-menu"
              style={contextMenuPosition}
              role="menu"
              aria-label="Annotation options"
            >
              <div className="annotation-context-title">Options</div>
              <div
                className={
                  contextMenuClassOptions.length > 0
                    ? 'annotation-context-branch'
                    : 'annotation-context-branch is-disabled'
                }
                onPointerEnter={() => {
                  if (contextMenuClassOptions.length > 0) {
                    openContextSubmenu()
                  }
                }}
                onPointerLeave={() => {
                  if (contextMenuClassOptions.length > 0) {
                    scheduleContextSubmenuClose()
                  }
                }}
              >
                <button
                  type="button"
                  className="annotation-context-action annotation-context-action-branch"
                  disabled={contextMenuClassOptions.length === 0}
                  aria-haspopup={contextMenuClassOptions.length > 0 ? 'menu' : undefined}
                  aria-expanded={contextMenuClassOptions.length > 0 ? isContextSubmenuOpen : undefined}
                  onClick={() => {
                    if (contextMenuClassOptions.length > 0) {
                      openContextSubmenu()
                    }
                  }}
                >
                  <span className="annotation-context-action-copy">
                    <span className="annotation-context-icon" aria-hidden="true">
                      <svg viewBox="0 0 16 16" fill="none">
                        <path
                          d="M3.25 4.5h6.5"
                          stroke="currentColor"
                          strokeWidth="1.35"
                          strokeLinecap="round"
                        />
                        <path
                          d="M3.25 8h9.5"
                          stroke="currentColor"
                          strokeWidth="1.35"
                          strokeLinecap="round"
                        />
                        <path
                          d="M3.25 11.5h6.5"
                          stroke="currentColor"
                          strokeWidth="1.35"
                          strokeLinecap="round"
                        />
                      </svg>
                    </span>
                    <span>Change class</span>
                  </span>
                  <span className="annotation-context-arrow" aria-hidden="true">
                    ›
                  </span>
                </button>
                {contextMenuClassOptions.length > 0 ? (
                  <div
                    className={contextMenuSubmenuClassName}
                    role="menu"
                    aria-label="Change class"
                    onPointerEnter={openContextSubmenu}
                    onPointerLeave={scheduleContextSubmenuClose}
                  >
                    <div className="annotation-context-title">Classes</div>
                    {contextMenuClassOptions.map((label) => (
                      <button
                        key={label}
                        type="button"
                        className="annotation-context-action"
                        onClick={() => {
                          onChangeAnnotationLabel(contextMenu.annotationId, label)
                          setContextMenu(null)
                        }}
                      >
                        <span>{label}</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
              <button
                type="button"
                className="annotation-context-action"
                onClick={() => {
                  onDeleteAnnotation(contextMenu.annotationId)
                  setContextMenu(null)
                }}
              >
                <span className="annotation-context-icon" aria-hidden="true">
                  <svg viewBox="0 0 16 16" fill="none">
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
                </span>
                <span>Delete box</span>
              </button>
              <button
                type="button"
                className="annotation-context-action"
                onClick={() => {
                  onDuplicateAnnotation(contextMenu.annotationId)
                  setContextMenu(null)
                }}
              >
                <span className="annotation-context-icon" aria-hidden="true">
                  <svg viewBox="0 0 16 16" fill="none">
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
                </span>
                <span>Duplicate box</span>
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  )
}, areAnnotationViewportPropsEqual)

function handlePointerMove(
  event: ReactPointerEvent<SVGSVGElement>,
  image: LoadedImage,
  imagePlacement: ContainRect,
  onUpdateDrawing: (point: Point) => void,
) {
  if (event.buttons !== 1) {
    return
  }

  const point = pointFromEvent(event, image, imagePlacement, 'clamp')
  if (!point) {
    return
  }

  onUpdateDrawing(point)
}

function handlePointerUp(
  event: ReactPointerEvent<SVGSVGElement>,
  image: LoadedImage,
  imagePlacement: ContainRect,
  onFinishDrawing: (point: Point) => void,
) {
  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
    event.currentTarget.releasePointerCapture(event.pointerId)
  }

  const point = pointFromEvent(event, image, imagePlacement, 'clamp')
  if (!point) {
    return
  }

  onFinishDrawing(point)
}

function pointFromEvent(
  event: ReactPointerEvent<SVGElement>,
  image: LoadedImage,
  imagePlacement: ContainRect,
  mode: 'strict' | 'clamp',
) {
  const bounds =
    event.currentTarget instanceof SVGSVGElement
      ? event.currentTarget.getBoundingClientRect()
      : event.currentTarget.ownerSVGElement?.getBoundingClientRect()

  if (!bounds) {
    return null
  }

  return pointFromClient(
    event.clientX,
    event.clientY,
    bounds,
    image,
    imagePlacement,
    mode,
  )
}

function stagePointFromEvent(event: ReactPointerEvent<SVGElement>) {
  const bounds =
    event.currentTarget instanceof SVGSVGElement
      ? event.currentTarget.getBoundingClientRect()
      : event.currentTarget.ownerSVGElement?.getBoundingClientRect()

  if (!bounds) {
    return null
  }

  return stagePointFromClient(event.clientX, event.clientY, bounds)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function withOpacity(color: string, opacity: number) {
  const hex = color.replace('#', '')

  if (hex.length !== 6) {
    return color
  }

  const red = Number.parseInt(hex.slice(0, 2), 16)
  const green = Number.parseInt(hex.slice(2, 4), 16)
  const blue = Number.parseInt(hex.slice(4, 6), 16)

  return `rgba(${red}, ${green}, ${blue}, ${opacity})`
}

function clampZoom(zoom: number) {
  return clamp(zoom, MIN_ZOOM, MAX_ZOOM)
}

function clampRectToImage(rect: Rect, image: LoadedImage): Rect {
  return {
    x: clamp(rect.x, 0, image.width - rect.width),
    y: clamp(rect.y, 0, image.height - rect.height),
    width: rect.width,
    height: rect.height,
  }
}

function resizeRectFromHandle(
  originRect: Rect,
  handle: ResizeHandle,
  point: Point,
  startPoint: Point,
  image: LoadedImage,
): Rect {
  const left = originRect.x
  const top = originRect.y
  const right = originRect.x + originRect.width
  const bottom = originRect.y + originRect.height
  const deltaX = point.x - startPoint.x
  const deltaY = point.y - startPoint.y

  let nextLeft = left
  let nextTop = top
  let nextRight = right
  let nextBottom = bottom

  if (handle.includes('w')) {
    nextLeft = clamp(left + deltaX, 0, right - MIN_ANNOTATION_SIZE)
  }

  if (handle.includes('e')) {
    nextRight = clamp(
      right + deltaX,
      left + MIN_ANNOTATION_SIZE,
      image.width,
    )
  }

  if (handle.includes('n')) {
    nextTop = clamp(top + deltaY, 0, bottom - MIN_ANNOTATION_SIZE)
  }

  if (handle.includes('s')) {
    nextBottom = clamp(
      bottom + deltaY,
      top + MIN_ANNOTATION_SIZE,
      image.height,
    )
  }

  return {
    x: nextLeft,
    y: nextTop,
    width: nextRight - nextLeft,
    height: nextBottom - nextTop,
  }
}

function buildResizeHandles(rect: Rect, effectiveScale: number) {
  const toStageUnits = (screenPx: number) => screenPx / effectiveScale
  const minSideScreenPx = Math.max(
    Math.min(rect.width, rect.height) * effectiveScale,
    0,
  )
  const radiusPx = clamp(minSideScreenPx * 0.1, 3.6, 7)
  const hitRadiusPx = clamp(Math.max(radiusPx + 3.8, 9.5), 9.5, 12)
  const strokeWidthPx = clamp(radiusPx * 0.34, 1.1, 1.9)
  const radius = toStageUnits(radiusPx)
  const hitRadius = toStageUnits(hitRadiusPx)
  const strokeWidth = toStageUnits(strokeWidthPx)

  return [
    { id: 'nw' as const, cx: rect.x, cy: rect.y, radius, hitRadius, strokeWidth },
    {
      id: 'ne' as const,
      cx: rect.x + rect.width,
      cy: rect.y,
      radius,
      hitRadius,
      strokeWidth,
    },
    {
      id: 'sw' as const,
      cx: rect.x,
      cy: rect.y + rect.height,
      radius,
      hitRadius,
      strokeWidth,
    },
    {
      id: 'se' as const,
      cx: rect.x + rect.width,
      cy: rect.y + rect.height,
      radius,
      hitRadius,
      strokeWidth,
    },
  ]
}

function pointFromClient(
  clientX: number,
  clientY: number,
  bounds: DOMRect,
  image: LoadedImage,
  imagePlacement: ContainRect,
  mode: 'strict' | 'clamp',
) {
  const stagePoint = stagePointFromClient(clientX, clientY, bounds, mode)
  if (!stagePoint) {
    return null
  }

  const x = (stagePoint.x - imagePlacement.x) / imagePlacement.scale
  const y = (stagePoint.y - imagePlacement.y) / imagePlacement.scale

  const insideImage =
    x >= 0 &&
    x <= image.width &&
    y >= 0 &&
    y <= image.height

  if (mode === 'strict' && !insideImage) {
    return null
  }

  return {
    x: clamp(x, 0, image.width),
    y: clamp(y, 0, image.height),
  }
}

function stagePointFromClient(
  clientX: number,
  clientY: number,
  bounds: DOMRect,
  mode: 'strict' | 'clamp' = 'strict',
) {
  const renderedStageRect = fitContainRect(
    STAGE_WIDTH,
    STAGE_HEIGHT,
    bounds.width,
    bounds.height,
  )
  const stageX =
    (clientX - bounds.left - renderedStageRect.x) / renderedStageRect.scale
  const stageY =
    (clientY - bounds.top - renderedStageRect.y) / renderedStageRect.scale
  const insideStage =
    stageX >= 0 &&
    stageX <= STAGE_WIDTH &&
    stageY >= 0 &&
    stageY <= STAGE_HEIGHT

  if (mode === 'strict' && !insideStage) {
    return null
  }

  return {
    x: clamp(stageX, 0, STAGE_WIDTH),
    y: clamp(stageY, 0, STAGE_HEIGHT),
  }
}

function getStageOffset(
  viewport: { width: number; height: number },
  zoom: number,
) {
  return {
    x: (viewport.width - viewport.width * zoom) / 2,
    y: (viewport.height - viewport.height * zoom) / 2,
  }
}

function clampPan(
  pan: Point,
  viewport: { width: number; height: number },
  zoom: number,
) {
  const offset = getStageOffset(viewport, zoom)
  const scaledWidth = viewport.width * zoom
  const scaledHeight = viewport.height * zoom
  const canPanX = scaledWidth > viewport.width
  const canPanY = scaledHeight > viewport.height

  const minPanX = canPanX ? viewport.width - scaledWidth - offset.x : 0
  const maxPanX = canPanX ? -offset.x : 0
  const minPanY = canPanY ? viewport.height - scaledHeight - offset.y : 0
  const maxPanY = canPanY ? -offset.y : 0

  return {
    x: clamp(pan.x, minPanX, maxPanX),
    y: clamp(pan.y, minPanY, maxPanY),
  }
}

type ContainRect = {
  x: number
  y: number
  width: number
  height: number
  scale: number
}

function fitContainRect(
  imageWidth: number,
  imageHeight: number,
  stageWidth: number,
  stageHeight: number,
): ContainRect {
  const scale = Math.min(stageWidth / imageWidth, stageHeight / imageHeight)
  const width = imageWidth * scale
  const height = imageHeight * scale

  return {
    x: (stageWidth - width) / 2,
    y: (stageHeight - height) / 2,
    width,
    height,
    scale,
  }
}

function getAutoFitZoom(imageWidth: number, imageHeight: number) {
  const placement = fitContainRect(
    imageWidth,
    imageHeight,
    STAGE_WIDTH,
    STAGE_HEIGHT,
  )

  return clampZoom(
    Math.max(
      STAGE_WIDTH / placement.width,
      STAGE_HEIGHT / placement.height,
    ),
  )
}

function mapAnnotationToStage(
  annotation: Annotation,
  image: LoadedImage,
  imagePlacement: ContainRect,
): Rect {
  const scaleX = imagePlacement.width / image.width
  const scaleY = imagePlacement.height / image.height

  return {
    x: imagePlacement.x + annotation.x * scaleX,
    y: imagePlacement.y + annotation.y * scaleY,
    width: annotation.width * scaleX,
    height: annotation.height * scaleY,
  }
}

function buildAnnotationGeometry(
  annotation: Annotation,
  image: LoadedImage,
  imagePlacement: ContainRect,
  overlayMetrics: OverlayMetrics,
): AnnotationGeometry {
  const stageRect = snapRectToStagePixels(
    mapAnnotationToStage(annotation, image, imagePlacement),
    overlayMetrics.effectiveScale,
  )
  const labelMetrics = getAnnotationLabelMetrics(stageRect, overlayMetrics)
  const labelWidth = snapStageValue(
    Math.max(
      labelMetrics.minTagWidth,
      Math.round(annotation.label.length * labelMetrics.characterWidth) +
        labelMetrics.tagPaddingX * 2,
    ),
    overlayMetrics.effectiveScale,
  )
  const labelX = snapStageValue(
    clamp(
      stageRect.x,
      overlayMetrics.edgeInset,
      Math.max(
        overlayMetrics.edgeInset,
        STAGE_WIDTH - overlayMetrics.edgeInset - labelWidth,
      ),
    ),
    overlayMetrics.effectiveScale,
  )
  const labelY = snapStageValue(
    Math.max(
      overlayMetrics.edgeInset,
      stageRect.y - labelMetrics.tagHeight - labelMetrics.labelGap,
    ),
    overlayMetrics.effectiveScale,
  )

  return {
    annotation,
    stageRect,
    labelRect: {
      x: labelX,
      y: labelY,
      width: labelWidth,
      height: labelMetrics.tagHeight,
    },
    labelMetrics,
  }
}

function getAnnotationCandidatesAtStagePoint(
  geometries: AnnotationGeometry[],
  point: Point,
  borderHitTolerance: number,
): Annotation[] {
  const labelHits = geometries
    .filter((geometry) => isPointWithinRect(geometry.labelRect, point))
    .map((geometry) => ({
      annotation: geometry.annotation,
      area: geometry.annotation.width * geometry.annotation.height,
      borderDistance: getRectBorderDistance(geometry.labelRect, point),
    }))

  if (labelHits.length > 0) {
    return labelHits
      .sort((left, right) => left.area - right.area)
      .map((candidate) => candidate.annotation)
  }

  const containingAnnotations = geometries
    .filter((geometry) => isPointWithinRect(geometry.stageRect, point))
    .map((geometry) => ({
      annotation: geometry.annotation,
      area: geometry.annotation.width * geometry.annotation.height,
      borderDistance: getRectBorderDistance(geometry.stageRect, point),
    }))

  if (containingAnnotations.length > 0) {
    return containingAnnotations
      .sort((left, right) =>
        left.area !== right.area
          ? left.area - right.area
          : left.borderDistance - right.borderDistance,
      )
      .map((candidate) => candidate.annotation)
  }

  const outsideBorderCandidates = geometries
    .map((geometry) => ({
      annotation: geometry.annotation,
      area: geometry.annotation.width * geometry.annotation.height,
      borderDistance: Math.min(
        getRectBorderDistance(geometry.stageRect, point),
        getRectBorderDistance(geometry.labelRect, point),
      ),
    }))
    .sort(
      (left, right) =>
        left.borderDistance !== right.borderDistance
          ? left.borderDistance - right.borderDistance
          : left.area - right.area,
    )
    .filter((candidate) => candidate.borderDistance <= borderHitTolerance)

  return outsideBorderCandidates.map((candidate) => candidate.annotation)
}

function layoutAnnotationLabels(
  geometries: AnnotationGeometry[],
  overlayMetrics: OverlayMetrics,
) {
  const positionedGeometries = [...geometries]
  const occupiedLabelRects: Rect[] = []
  const placementOrder = geometries
    .map((geometry, index) => ({
      geometry,
      index,
      area: geometry.annotation.width * geometry.annotation.height,
    }))
    .sort((left, right) =>
      left.geometry.labelRect.y !== right.geometry.labelRect.y
        ? left.geometry.labelRect.y - right.geometry.labelRect.y
        : left.geometry.labelRect.x !== right.geometry.labelRect.x
          ? left.geometry.labelRect.x - right.geometry.labelRect.x
          : left.area - right.area,
    )

  for (const { geometry, index } of placementOrder) {
    const labelRect = placeLabelRect(geometry, occupiedLabelRects, overlayMetrics)
    positionedGeometries[index] = {
      ...geometry,
      labelRect,
    }
    occupiedLabelRects.push(labelRect)
  }

  return positionedGeometries
}

function placeLabelRect(
  geometry: AnnotationGeometry,
  occupiedLabelRects: Rect[],
  overlayMetrics: OverlayMetrics,
) {
  const slotStep = geometry.labelRect.height + geometry.labelMetrics.labelGap
  const minY = overlayMetrics.edgeInset
  const maxY = STAGE_HEIGHT - overlayMetrics.edgeInset - geometry.labelRect.height
  const aboveY = clamp(
    geometry.stageRect.y - geometry.labelRect.height - geometry.labelMetrics.labelGap,
    minY,
    maxY,
  )
  const belowY = clamp(
    geometry.stageRect.y + geometry.labelMetrics.labelGap,
    minY,
    maxY,
  )
  const candidateRects: Rect[] = []
  const seenKeys = new Set<string>()
  const maxRows = Math.max(1, Math.ceil(STAGE_HEIGHT / Math.max(slotStep, 1)) + 1)

  const addCandidate = (y: number) => {
    const clampedY = snapStageValue(clamp(y, minY, maxY), overlayMetrics.effectiveScale)
    const key = `${geometry.labelRect.x}:${clampedY}`
    if (seenKeys.has(key)) {
      return
    }
    seenKeys.add(key)
    candidateRects.push({
      ...geometry.labelRect,
      y: clampedY,
    })
  }

  for (let level = 0; level < maxRows; level += 1) {
    addCandidate(aboveY - slotStep * level)
  }

  for (let level = 0; level < maxRows; level += 1) {
    addCandidate(belowY + slotStep * level)
  }

  const freeCandidate = candidateRects.find(
    (candidate) =>
      !occupiedLabelRects.some((occupiedRect) =>
        rectsOverlap(candidate, occupiedRect),
      ),
  )
  if (freeCandidate) {
    return freeCandidate
  }

  return (
    candidateRects.reduce((bestRect, candidate) => {
      const overlapScore = occupiedLabelRects.reduce(
        (total, occupiedRect) => total + getRectOverlapArea(candidate, occupiedRect),
        0,
      )
      const bestScore = occupiedLabelRects.reduce(
        (total, occupiedRect) => total + getRectOverlapArea(bestRect, occupiedRect),
        0,
      )

      return overlapScore < bestScore ? candidate : bestRect
    }) ?? geometry.labelRect
  )
}

function resolveAnnotationAtPoint(
  candidates: Annotation[],
  point: Point,
  selectedId: string | null,
  previousCycle:
    | {
        point: Point
        candidateIds: string[]
      }
    | null,
) {
  if (candidates.length === 0) {
    return null
  }

  const candidateIds = candidates.map((candidate) => candidate.id)
  const canCycle =
    previousCycle !== null &&
    areCandidateListsEqual(previousCycle.candidateIds, candidateIds) &&
    getPointDistance(previousCycle.point, point) <= 6

  if (!canCycle || !selectedId) {
    return candidates[0] ?? null
  }

  const currentIndex = candidates.findIndex(
    (candidate) => candidate.id === selectedId,
  )
  if (currentIndex < 0) {
    return candidates[0] ?? null
  }

  return candidates[(currentIndex + 1) % candidates.length] ?? null
}

function getRectBorderDistance(rect: Rect, point: Point) {
  const left = rect.x
  const top = rect.y
  const right = rect.x + rect.width
  const bottom = rect.y + rect.height

  const dx =
    point.x < left ? left - point.x : point.x > right ? point.x - right : 0
  const dy =
    point.y < top ? top - point.y : point.y > bottom ? point.y - bottom : 0

  if (dx > 0 || dy > 0) {
    return Math.hypot(dx, dy)
  }

  return Math.min(
    point.x - left,
    right - point.x,
    point.y - top,
    bottom - point.y,
  )
}

function isPointWithinRect(rect: Rect, point: Point) {
  return (
    point.x >= rect.x &&
    point.x <= rect.x + rect.width &&
    point.y >= rect.y &&
    point.y <= rect.y + rect.height
  )
}

function rectsOverlap(left: Rect, right: Rect) {
  return !(
    left.x + left.width <= right.x ||
    right.x + right.width <= left.x ||
    left.y + left.height <= right.y ||
    right.y + right.height <= left.y
  )
}

function getRectOverlapArea(left: Rect, right: Rect) {
  const overlapWidth =
    Math.min(left.x + left.width, right.x + right.width) -
    Math.max(left.x, right.x)
  const overlapHeight =
    Math.min(left.y + left.height, right.y + right.height) -
    Math.max(left.y, right.y)

  if (overlapWidth <= 0 || overlapHeight <= 0) {
    return 0
  }

  return overlapWidth * overlapHeight
}

function getLabelConnector(labelRect: Rect, stageRect: Rect) {
  const labelCenterX = labelRect.x + labelRect.width / 2
  const labelCenterY = labelRect.y + labelRect.height / 2
  const boxCenterX = stageRect.x + stageRect.width / 2
  const boxCenterY = stageRect.y + stageRect.height / 2

  if (labelRect.y + labelRect.height <= stageRect.y) {
    return {
      start: {
        x: labelCenterX,
        y: labelRect.y + labelRect.height,
      },
      end: {
        x: boxCenterX,
        y: stageRect.y,
      },
    }
  }

  if (labelRect.y >= stageRect.y + stageRect.height) {
    return {
      start: {
        x: labelCenterX,
        y: labelRect.y,
      },
      end: {
        x: boxCenterX,
        y: stageRect.y + stageRect.height,
      },
    }
  }

  if (labelRect.x + labelRect.width <= stageRect.x) {
    return {
      start: {
        x: labelRect.x + labelRect.width,
        y: labelCenterY,
      },
      end: {
        x: stageRect.x,
        y: boxCenterY,
      },
    }
  }

  if (labelRect.x >= stageRect.x + stageRect.width) {
    return {
      start: {
        x: labelRect.x,
        y: labelCenterY,
      },
      end: {
        x: stageRect.x + stageRect.width,
        y: boxCenterY,
      },
    }
  }

  return {
    start: {
      x: labelCenterX,
      y: labelCenterY,
    },
    end: {
      x: boxCenterX,
      y: boxCenterY,
    },
  }
}

function getPointDistance(left: Point, right: Point) {
  return Math.hypot(left.x - right.x, left.y - right.y)
}

function areCandidateListsEqual(left: string[], right: string[]) {
  if (left.length !== right.length) {
    return false
  }

  return left.every((candidateId, index) => candidateId === right[index])
}

function getBorderHitTolerance(shellWidth: number) {
  const screenPixelsPerStageUnit = shellWidth / STAGE_WIDTH

  if (!Number.isFinite(screenPixelsPerStageUnit) || screenPixelsPerStageUnit <= 0) {
    return 16
  }

  return clamp(12 / screenPixelsPerStageUnit, 4, 24)
}

type OverlayMetrics = {
  effectiveScale: number
  fontSize: number
  textStrokeWidth: number
  characterWidth: number
  tagHeight: number
  tagPaddingX: number
  tagRadius: number
  labelGap: number
  edgeInset: number
  minTagWidth: number
}

function getAnnotationLabelMetrics(
  stageRect: Rect,
  overlayMetrics: OverlayMetrics,
): AnnotationLabelMetrics {
  const minSideScreenPx = Math.max(
    Math.min(stageRect.width, stageRect.height) * overlayMetrics.effectiveScale,
    0,
  )
  const compactScale = clamp(0.54 + ((minSideScreenPx - 18) / 54) * 0.46, 0.54, 1)
  const toStageUnits = (screenPx: number) => screenPx / overlayMetrics.effectiveScale
  const minFontSize = toStageUnits(10)
  const minTextStrokeWidth = toStageUnits(1.1)
  const minTagHeight = toStageUnits(19)
  const minTagPaddingX = toStageUnits(6)
  const minTagRadius = toStageUnits(2.2)
  const minLabelGap = toStageUnits(4)
  const minTagWidth = toStageUnits(44)

  return {
    fontSize: Math.max(overlayMetrics.fontSize * compactScale, minFontSize),
    textStrokeWidth: Math.max(
      overlayMetrics.textStrokeWidth * compactScale,
      minTextStrokeWidth,
    ),
    characterWidth: Math.max(
      overlayMetrics.characterWidth * compactScale,
      minFontSize * 0.62,
    ),
    tagHeight: Math.max(overlayMetrics.tagHeight * compactScale, minTagHeight),
    tagPaddingX: Math.max(
      overlayMetrics.tagPaddingX * compactScale,
      minTagPaddingX,
    ),
    tagRadius: Math.max(overlayMetrics.tagRadius * compactScale, minTagRadius),
    labelGap: Math.max(overlayMetrics.labelGap * compactScale, minLabelGap),
    minTagWidth: Math.max(overlayMetrics.minTagWidth * compactScale, minTagWidth),
  }
}

function getOverlayMetrics(
  viewport: { width: number; height: number },
  zoom: number,
): OverlayMetrics {
  const effectiveScale = getOverlayEffectiveScale(viewport, zoom)
  const toStageUnits = (screenPx: number) => screenPx / effectiveScale
  const zoomVisualScale = clamp(Math.pow(zoom, 0.45), 1, 1.9)

  const fontSizePx = 13 * zoomVisualScale
  const tagPaddingXPx = 10 * zoomVisualScale
  const tagHeightPx = 24 * zoomVisualScale
  const textStrokeWidthPx = clamp(1.6 * zoomVisualScale, 1.6, 2.7)
  const minTagWidthPx = 56 * zoomVisualScale
  const tagRadiusPx = clamp(2 * zoomVisualScale, 2, 4)
  const labelGapPx = 6 * clamp(Math.pow(zoom, 0.3), 1, 1.45)
  const edgeInsetPx = 4

  return {
    effectiveScale,
    fontSize: toStageUnits(fontSizePx),
    textStrokeWidth: toStageUnits(textStrokeWidthPx),
    characterWidth: toStageUnits(fontSizePx) * 0.62,
    tagHeight: toStageUnits(tagHeightPx),
    tagPaddingX: toStageUnits(tagPaddingXPx),
    tagRadius: toStageUnits(tagRadiusPx),
    labelGap: toStageUnits(labelGapPx),
    edgeInset: toStageUnits(edgeInsetPx),
    minTagWidth: toStageUnits(minTagWidthPx),
  }
}

function getOverlayEffectiveScale(
  viewport: { width: number; height: number },
  zoom: number,
) {
  return Math.max(
    Math.min(viewport.width / STAGE_WIDTH, viewport.height / STAGE_HEIGHT) * zoom,
    0.0001,
  )
}

function snapCssPixel(value: number) {
  if (typeof window === 'undefined') {
    return Math.round(value)
  }

  const dpr = window.devicePixelRatio || 1
  return Math.round(value * dpr) / dpr
}

function snapStageValue(value: number, effectiveScale: number) {
  if (effectiveScale <= 0) {
    return value
  }

  return Math.round(value * effectiveScale) / effectiveScale
}

function snapRectToStagePixels(rect: Rect, effectiveScale: number): Rect {
  const minDimension = 1 / effectiveScale
  return {
    x: snapStageValue(rect.x, effectiveScale),
    y: snapStageValue(rect.y, effectiveScale),
    width: Math.max(snapStageValue(rect.width, effectiveScale), minDimension),
    height: Math.max(snapStageValue(rect.height, effectiveScale), minDimension),
  }
}

function areAnnotationViewportPropsEqual(
  previousProps: Readonly<AnnotationViewportProps>,
  nextProps: Readonly<AnnotationViewportProps>,
) {
  return (
    areLoadedImagesEqual(previousProps.image, nextProps.image) &&
    previousProps.imageLabel === nextProps.imageLabel &&
    previousProps.isLoading === nextProps.isLoading &&
    previousProps.isError === nextProps.isError &&
    previousProps.annotations === nextProps.annotations &&
    previousProps.selectedId === nextProps.selectedId &&
    areRectsEqual(previousProps.draftRect, nextProps.draftRect) &&
    previousProps.tool === nextProps.tool &&
    previousProps.showSamTools === nextProps.showSamTools &&
    previousProps.isSamBusy === nextProps.isSamBusy &&
    previousProps.recentDatasets === nextProps.recentDatasets &&
    previousProps.openDatasetDisabled === nextProps.openDatasetDisabled &&
    previousProps.onSelectTool === nextProps.onSelectTool &&
    previousProps.onOpenDataset === nextProps.onOpenDataset &&
    previousProps.onOpenRecentDataset === nextProps.onOpenRecentDataset &&
    previousProps.onRemoveRecentDataset === nextProps.onRemoveRecentDataset &&
    previousProps.onStartDrawing === nextProps.onStartDrawing &&
    previousProps.onUpdateDrawing === nextProps.onUpdateDrawing &&
    previousProps.onFinishDrawing === nextProps.onFinishDrawing &&
    areStringArraysEqual(previousProps.classOptions, nextProps.classOptions) &&
    previousProps.onSelectAnnotation === nextProps.onSelectAnnotation &&
    previousProps.onUpdateAnnotationRect === nextProps.onUpdateAnnotationRect &&
    previousProps.onChangeAnnotationLabel === nextProps.onChangeAnnotationLabel &&
    previousProps.onDuplicateAnnotation === nextProps.onDuplicateAnnotation &&
    previousProps.onDeleteAnnotation === nextProps.onDeleteAnnotation &&
    previousProps.onHoverPointChange === nextProps.onHoverPointChange
  )
}

function areLoadedImagesEqual(
  left: LoadedImage | null,
  right: LoadedImage | null,
) {
  if (left === right) {
    return true
  }

  if (!left || !right) {
    return false
  }

  return (
    left.id === right.id &&
    left.url === right.url &&
    left.width === right.width &&
    left.height === right.height
  )
}

function areStringArraysEqual(left: string[], right: string[]) {
  if (left === right) {
    return true
  }

  if (left.length !== right.length) {
    return false
  }

  return left.every((value, index) => value === right[index])
}

function areRectsEqual(left: Rect | null, right: Rect | null) {
  if (left === right) {
    return true
  }

  if (!left || !right) {
    return false
  }

  return (
    left.x === right.x &&
    left.y === right.y &&
    left.width === right.width &&
    left.height === right.height
  )
}
