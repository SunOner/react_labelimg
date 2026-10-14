import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
import { appHandler } from './appHandler.mjs'

const source = fs.readFileSync(
  new URL('../src/lib/video.ts', import.meta.url),
  'utf8',
)
const code = ts.transpileModule(source, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
  },
}).outputText
const {
  boundedVideoTime,
  videoFrameFilename,
  videoFrameKey,
  identifyVideo,
  captureVideoFrame,
  formatVideoTime,
} = await import(`data:text/javascript,${encodeURIComponent(code)}`)

test('source sampling is bounded and frame names distinguish sources and decoded timestamps', async () => {
  const file = new File(['test-video'], 'ролик.mp4')
  const id = await identifyVideo(file)
  assert.equal(await identifyVideo(file), id)
  assert.notEqual(
    await identifyVideo(new File(['different-video'], 'ролик.mp4')),
    id,
  )
  const first = videoFrameFilename(file.name, id, 1.04, 'png')
  assert.equal(first, `ролик_${id}_t000001040000us.png`)
  assert.notEqual(first, videoFrameFilename(file.name, id, 1.08, 'png'))
  const safe = videoFrameFilename('a/b:c.mp4', id, 0, 'jpg')
  assert.ok(!safe.includes('/') && !safe.includes(':'))
  let sampledBytes = 0
  const largeFile = {
    name: 'large.mp4',
    size: 1_000_000_000,
    slice(start, end = 1_000_000_000) {
      sampledBytes += end - start
      return new Blob(['sample'])
    },
  }
  await identifyVideo(largeFile)
  assert.equal(sampledBytes, 131072)
})

test('time stepping advances the requested time even when multiple seeks decode the same frame', () => {
  const requestedTimes = [0, 1 / 30, 2 / 30, 3 / 30]
  assert.deepEqual(
    requestedTimes.map((time) => boundedVideoTime(time, 4)),
    requestedTimes,
  )
  assert.equal(boundedVideoTime(-1, 4), 0)
  assert.ok(boundedVideoTime(5, 4) < 4)
  assert.equal(formatVideoTime(3661.04), '01:01:01.040')
})

test('capture encodes native video dimensions and refuses seeking or undecoded media', async () => {
  const drawn = []
  const canvas = {
    width: 0,
    height: 0,
    getContext() {
      return {
        drawImage(...args) {
          drawn.push(args)
        },
      }
    },
    toBlob(callback, mime, quality) {
      assert.equal(this.width, 1920)
      assert.equal(this.height, 1080)
      assert.equal(mime, 'image/jpeg')
      assert.equal(quality, 0.95)
      callback(new Blob(['frame'], { type: mime }))
    },
  }
  const oldDocument = globalThis.document
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, 'canvas')
      return canvas
    },
  }
  try {
    const video = {
      seeking: false,
      readyState: 2,
      videoWidth: 1920,
      videoHeight: 1080,
    }
    const blob = await captureVideoFrame(video, 'jpg')
    assert.equal(blob.type, 'image/jpeg')
    assert.deepEqual(drawn[0], [video, 0, 0, 1920, 1080])
    assert.equal(canvas.width, 0)
    await assert.rejects(
      captureVideoFrame({ ...video, seeking: true }, 'png'),
      /finish loading/,
    )
    await assert.rejects(
      captureVideoFrame({ ...video, readyState: 1 }, 'png'),
      /finish loading/,
    )
  } finally {
    globalThis.document = oldDocument
  }
})

function saveContext() {
  const context = {
    currentSessionId: 'session',
    isDatasetSession: true,
    currentSessionRootPath: '/dataset',
    currentSessionIdRef: { current: 'session' },
    projectClasses: ['existing'],
    imageIdSetRef: { current: new Set(['old']) },
    annotationsByImageRef: { current: {} },
    images: [{ id: 'old', relativePath: 'original.png' }],
    classes: {},
    currentEntry: null,
    states: {},
    info: 'old',
    sanitizeClassList: (labels) => [
      ...new Set(labels.map((label) => label.trim())),
    ],
    labelToColor: () => '#123456',
    buildLocalImageUrl: (session, id) => `/api/${session}/${id}`,
    setProjectClassesByRootPath(update) {
      context.classes = update(context.classes)
    },
    setImages(update) {
      context.images = update(context.images)
    },
    commitAnnotationsByImage(update) {
      context.annotationsByImageRef.current = update(
        context.annotationsByImageRef.current,
      )
    },
    setAnnotationLoadState(id, state) {
      context.states[id] = state
    },
    setCurrentImageEntry(update) {
      context.currentEntry = update(context.currentEntry)
    },
    setProjectInfo(info) {
      context.info = info
    },
    setProjectInfoStatus() {},
  }
  return context
}

test('video opens during indexing once the target dataset session is published', async () => {
  const context = {
    isDatasetSession: true, currentSessionId: 'dataset', backendStatus: 'online',
    isDropUploadBusy: false, isOpeningSession: true, loadingSessionId: 'dataset',
  }
  assert.equal(appHandler('canOpenVideo', context), true)
  assert.equal(appHandler('canOpenVideo', { ...context, loadingSessionId: null }), false)
  assert.equal(appHandler('canOpenVideo', { ...context, loadingSessionId: 'other' }), false)
  assert.equal(appHandler('canOpenVideo', { ...context, backendStatus: 'offline' }), false)
  const events = []
  await appHandler('handleOpenVideoWorkspace', {
    ...context, canOpenVideo: true, setOpenMenu() {},
    setSessionError(value) { events.push(value) },
    async flushPendingAnnotationSaves() { events.push('flushed') },
    setVideoInitialFile() {}, setIsVideoWorkbenchOpen(value) { events.push(value) },
  })()
  assert.deepEqual(events, ['flushed', null, true])
})

test('stale directory snapshots retain newly saved frames without duplicate entries', () => {
  const merge = appHandler('mergeLoadingSessionImages', {})
  const saved = { id: 'frame', relativePath: 'clip.png', annotationCount: 2 }
  const snapshot = [{ id: 'old', relativePath: 'original.png', annotationCount: 0 }]
  assert.deepEqual(merge(snapshot, [saved]), [saved, ...snapshot])
  const complete = [...snapshot, { ...saved, annotationCount: 3 }]
  assert.deepEqual(merge(complete, [saved]), [{ ...saved, annotationCount: 3 }, ...snapshot])
})

test('a failed frame save leaves the App dataset and annotations intact', async () => {
  const context = saveContext()
  context.saveLocalVideoFrame = async () => {
    throw new Error('Disk full')
  }
  const save = appHandler('handleSaveVideoFrame', context)
  await assert.rejects(
    save(new File(['frame'], 'frame.png'), [{ label: 'cat' }], null),
    /Disk full/,
  )
  assert.equal(context.images.length, 1)
  assert.equal(context.info, 'old')
  assert.deepEqual(context.classes, {})
  assert.deepEqual(context.annotationsByImageRef.current, {})
})

test('frame save merges the canonical saved annotations and class names without changing selection', async () => {
  const context = saveContext()
  const entry = {
    id: 'frame',
    name: 'frame.png',
    relativePath: 'frame.png',
    annotationCount: 1,
  }
  const canonical = [
    {
      id: 'box',
      label: 'cat',
      x: 1,
      y: 2,
      width: 3,
      height: 4,
      difficult: false,
    },
  ]
  context.currentEntry = context.images[0]
  context.saveLocalVideoFrame = async (_session, _file, _boxes, classes) => {
    assert.deepEqual(classes, ['existing', 'cat'])
    return { image: entry, annotations: canonical }
  }
  const save = appHandler('handleSaveVideoFrame', context)
  const result = await save(
    new File(['frame'], 'frame.png'),
    [{ label: 'cat' }, { label: 'cat' }],
    null,
  )
  assert.equal(result.annotations.length, 1)
  assert.equal(context.images.length, 2)
  assert.equal(context.currentEntry.id, 'old')
  assert.equal(context.annotationsByImageRef.current.frame[0].color, '#123456')
  assert.equal(context.states.frame, 'ready')
  assert.deepEqual(context.classes['/dataset'], ['existing', 'cat'])
})

test('revisiting a saved frame updates annotations without uploading another image', async () => {
  const context = saveContext()
  const entry = {
    id: 'frame',
    name: 'frame.png',
    relativePath: 'frame.png',
    annotationCount: 0,
  }
  context.images.push(entry)
  context.saveLocalVideoFrame = async () =>
    assert.fail('Existing frame must not be uploaded again')
  context.saveLocalAnnotations = async (sessionId, imageId) => {
    assert.equal(sessionId, 'session')
    assert.equal(imageId, 'frame')
    return { count: 1, format: 'yolo' }
  }
  const save = appHandler('handleSaveVideoFrame', context)
  await save(new File(['frame'], 'frame.png'), [{ label: 'cat' }], entry)
  assert.equal(context.images.length, 2)
  assert.equal(
    context.images.find((image) => image.id === 'frame').annotationCount,
    1,
  )
})

const workbenchUrl = new URL(
  '../src/components/VideoWorkbench.tsx',
  import.meta.url,
)

function frameSaveContext(boxes = []) {
  const context = {
    frame: { blob: new Blob(['frame'], { type: 'image/png' }), filename: 'frame.png', time: 1, savedImage: null },
    source: { id: 'video' }, annotations: boxes, operationRef: { current: false },
    drawStartRef: { current: null }, baselineAnnotationsRef: { current: [] },
    annotationsRef: { current: boxes },
    dirtyRef: { current: true }, savedFrames: {}, calls: [], error: null,
    setError(error) { context.error = error }, setBusy() {},
    setFrame(frame) { context.frame = frame },
    setSavedFrames(update) { context.savedFrames = update(context.savedFrames) },
    setAnnotations(boxes) { context.annotations = boxes },
    setLastBoxes() {}, setDirty() {}, setMessage() {},
    videoFrameKey,
    async onSave(_file, annotations, existing) {
      context.calls.push({ annotations, existing })
      return { image: { id: 'frame', name: 'frame.png' }, annotations }
    },
  }
  return context
}

test('empty frame is saved only by an explicit background action, including cleared saved boxes', async () => {
  const context = frameSaveContext()
  const save = appHandler('save', context, workbenchUrl)
  assert.equal(await save(), false)
  assert.match(context.error, /Save as background/)
  assert.equal(context.calls.length, 0)
  assert.equal(await save('background'), true)
  assert.equal(context.calls.length, 1)
  assert.deepEqual(context.calls[0].annotations, [])
  assert.equal(await save(), false)
  assert.equal(context.calls.length, 1)
  assert.equal(await save('background'), true)
  assert.equal(context.calls[1].existing.id, 'frame')
})

test('background action never strips annotated boxes and normal save still works', async () => {
  const context = frameSaveContext([{ id: 'box', label: 'cat' }])
  const save = appHandler('save', context, workbenchUrl)
  assert.equal(await save('background'), false)
  assert.equal(context.calls.length, 0)
  assert.equal(context.annotations[0].label, 'cat')
  assert.equal(await save(), true)
  assert.equal(context.calls[0].annotations[0].label, 'cat')
})

test('empty frame cannot be implicitly saved when continuing navigation', async () => {
  const context = frameSaveContext()
  context.save = appHandler('save', context, workbenchUrl)
  let continued = false
  context.pendingActionRef = { current: () => { continued = true } }
  context.setConfirmNavigation = () => {}
  const next = appHandler('continueNavigation', context, workbenchUrl)
  await next(true)
  assert.equal(continued, false)
  assert.equal(context.calls.length, 0)
  await next(true, 'background')
  assert.equal(continued, true)
  assert.equal(context.calls.length, 1)
})
function progressContext() {
  const context = {
    source: { id: 'source' },
    metadataReady: true,
    sessionId: 'dataset',
    state: {
      positionSeconds: 10,
      durationSeconds: 60,
      width: 1920,
      height: 1080,
      stepFps: 25,
      sampleIntervalSeconds: 2,
      imageFormat: 'png',
      playbackRate: 1,
    },
    progressRequestRef: { current: null },
    lastProgressRef: { current: null },
    error: null,
    remembered: null,
    progressSnapshot() {
      return { ...context.state }
    },
    setMetadataError(error) {
      context.error = error
    },
    rememberVideo(video) {
      context.remembered = video
    },
  }
  return context
}

test('failed metadata writes can be retried without losing progress or a saved frame', async () => {
  const context = progressContext()
  context.saveVideoProgress = async () => {
    throw new Error('metadata storage unavailable')
  }
  const persist = appHandler('persistProgress', context, workbenchUrl)
  assert.equal(await persist(), false)
  assert.match(context.error, /storage unavailable/)
  assert.equal(context.lastProgressRef.current, null)
  context.saveVideoProgress = async (_session, _id, state) => ({
    id: 'source',
    state,
    frameCount: 7,
  })
  assert.equal(await persist(true), true)
  assert.equal(context.remembered.frameCount, 7)
  assert.equal(context.remembered.state.positionSeconds, 10)
  assert.equal(context.error, null)
})

test('close flush waits for an in-flight metadata write and saves the latest position', async () => {
  const context = progressContext()
  const positions = []
  let resolveFirst
  context.saveVideoProgress = async (_session, _id, state) => {
    positions.push(state.positionSeconds)
    if (positions.length === 1)
      await new Promise((resolve) => {
        resolveFirst = resolve
      })
    return { id: 'source', state, frameCount: 1 }
  }
  const persist = appHandler('persistProgress', context, workbenchUrl)
  const first = persist()
  context.state.positionSeconds = 12
  const closeFlush = persist(true)
  assert.deepEqual(positions, [10])
  resolveFirst()
  await Promise.all([first, closeFlush])
  assert.deepEqual(positions, [10, 12])
  assert.equal(context.remembered.state.positionSeconds, 12)
})

test('metadata progress is not overwritten before source resume and video metadata finish loading', async () => {
  const context = progressContext()
  context.metadataReady = false
  context.saveVideoProgress = async () =>
    assert.fail('Unloaded video must not overwrite saved progress with zero')
  const persist = appHandler('persistProgress', context, workbenchUrl)
  assert.equal(await persist(), true)
})

test('new frame upload carries its source video and timestamp to the backend transaction', async () => {
  const context = saveContext()
  const video = { videoId: '0123456789abcdef', timeSeconds: 1.04 }
  context.saveLocalVideoFrame = async (
    _session,
    _file,
    _boxes,
    _classes,
    provenance,
  ) => {
    assert.deepEqual(provenance, video)
    return {
      image: {
        id: 'frame',
        name: 'frame.png',
        relativePath: 'frame.png',
        annotationCount: 0,
      },
      annotations: [],
    }
  }
  await appHandler('handleSaveVideoFrame', context)(
    new File(['frame'], 'frame.png'),
    [],
    null,
    video,
  )
})

test('discard restores saved boxes even when opening the next source fails', async () => {
  const baseline = [{ id: 'box', label: 'saved class' }]
  const context = {
    annotations: [{ id: 'box', label: 'unsaved class' }],
    baselineAnnotationsRef: { current: baseline },
    annotationsRef: { current: baseline },
    annotationHistory: { clear() {} },
    setHistoryAvailability() {},
    dirtyRef: { current: true },
    pendingActionRef: {
      current: () => {
        context.error = 'Original source unavailable'
      },
    },
    save: async () => assert.fail('Discard must not save'),
    setAnnotations(value) {
      context.annotations = value
    },
    setSelectedId() {},
    setMessage() {},
    setConfirmNavigation() {},
    setDirty(value) {
      context.dirty = value
    },
  }
  await appHandler('continueNavigation', context, workbenchUrl)(false)
  assert.equal(context.annotations[0].label, 'saved class')
  assert.equal(context.dirty, false)
  assert.equal(context.error, 'Original source unavailable')
})

test('failed save-and-continue leaves the current annotations and navigation pending', async () => {
  const context = {
    save: async () => false,
    pendingActionRef: {
      current: () =>
        assert.fail('Navigation must remain blocked after a failed save'),
    },
    setAnnotations() {
      assert.fail('Failed save must not discard annotations')
    },
    setConfirmNavigation() {
      assert.fail('Failed save must keep the choice open')
    },
  }
  await appHandler('continueNavigation', context, workbenchUrl)(true)
  assert.ok(context.pendingActionRef.current)
})
