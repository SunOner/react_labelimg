import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
import { appHandler, appEffect } from './appHandler.mjs'

async function loadLibrary(name) {
  const source = fs.readFileSync(new URL(`../src/lib/${name}.ts`, import.meta.url), 'utf8')
    .replace("import.meta.env.VITE_API_BASE_URL ?? ''", "''")
  const code = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText
  return import(`data:text/javascript,${encodeURIComponent(code)}`)
}
const { trimAnnotationCache, summarizeAnnotations, aggregateAnnotationSummaries, renameAnnotationSummaries } = await loadLibrary('sessionMemory')
const { createImageResource } = await loadLibrary('imageResource')
const ref = current => ({ current })

test('browsing 2000 images bounds the actual App cache and keeps dirty/in-flight/SAM images', () => {
  const context = {
    annotationsByImageRef: ref({}), annotationSummariesRef: ref(new Map()),
    imageIdSetRef: ref(new Set(Array.from({ length: 2000 }, (_, i) => String(i)))),
    retainedAnnotationIdsRef: ref(new Set()), samImageIdsRef: ref(new Set(['2'])),
    queuedAnnotationSavesRef: ref({ '0': {} }), annotationSavePromisesRef: ref({ '1': {} }),
    annotationLoadStateRef: ref({}), annotationSaveRevisionRef: ref({}),
    currentEntry: null, errors: {}, states: {},
    trimAnnotationCache, summarizeAnnotations,
    setAnnotationSummaries() {}, setAnnotationsByImage() {},
    setAnnotationLoadStates(update) { context.states = update(context.states) },
    setAnnotationLoadErrors(update) { context.errors = update(context.errors) },
  }
  const commit = appHandler('commitAnnotationsByImage', context)
  for (let i = 0; i < 2000; i++) {
    const id = String(i)
    context.currentEntry = { id }
    context.retainedAnnotationIdsRef.current = new Set([id])
    context.annotationLoadStateRef.current[id] = 'ready'
    context.states[id] = 'ready'
    context.annotationSaveRevisionRef.current[id] = 1
    commit(cache => ({ ...cache, [id]: Array.from({ length: 50 }, (_, j) => ({
      id: `${id}-${j}`, label: 'cat', difficult: j === 0, x: j, y: j, width: 3, height: 3,
    })) }))
    assert.ok(Object.keys(context.annotationsByImageRef.current).length <= 24)
    assert.ok(Object.keys(context.annotationLoadStateRef.current).length <= 24)
  }
  for (const id of ['0', '1', '2', '1999']) assert.equal(context.annotationsByImageRef.current[id].length, 50)
  const stats = aggregateAnnotationSummaries(context.annotationSummariesRef.current.values(), g => g.label, () => true)
  assert.deepEqual(stats, { classCounts: { cat: 100000 }, imageCounts: { cat: 2000 }, difficultCount: 2000, unknownCount: 0 })
  assert.equal(context.annotationSummariesRef.current.get('50').length, 1)
  assert.equal('x' in context.annotationSummariesRef.current.get('50')[0], false)
  renameAnnotationSummaries(context.annotationSummariesRef.current, 'cat', 'kitten')
  assert.equal(context.annotationSummariesRef.current.get('50')[0].label, 'kitten')
  context.imageIdSetRef.current.delete('50')
  commit(cache => cache)
  assert.equal(context.annotationSummariesRef.current.has('50'), false)
})

test('annotation count budget evicts clean data while protecting an oversized current image', () => {
  const cache = { old: Array(19000), neighbor: Array(3000), current: Array(40000), dirty: Array(5) }
  const result = trimAnnotationCache(cache, ['current'], new Set(['current', 'dirty']))
  assert.deepEqual(Object.keys(result), ['current', 'dirty'])
  assert.equal(result.current, cache.current)
  assert.equal(result.dirty, cache.dirty)
})

test('current-image URLs are released across 500 downloads and repeated disposal is safe', async () => {
  const live = new Set()
  let created = 0
  let revoked = 0
  const io = {
    fetch: async () => new Response(new Uint8Array([1, 2, 3])),
    create() { const url = `blob:${++created}`; live.add(url); return url },
    revoke(url) { assert.ok(live.delete(url)); revoked++ },
  }
  for (let i = 0; i < 500; i++) {
    const resource = createImageResource('/same-image', io)
    assert.ok(live.has(await resource.promise))
    assert.equal(live.size, 1)
    resource.dispose()
    resource.dispose()
    assert.equal(resource.isDisposed(), true)
    assert.equal(live.size, 0)
  }
  assert.equal(created, revoked)
})

test('navigating away during download aborts it and cannot allocate a late object URL', async () => {
  let finish
  let signal
  const resource = createImageResource('/slow', {
    fetch: (_url, init) => { signal = init.signal; return new Promise(resolve => { finish = resolve }) },
    create() { throw Error('late response must not create an object URL') },
    revoke() { throw Error('no object URL was created') },
  })
  resource.dispose()
  assert.equal(signal.aborted, true)
  finish(new Response('bytes'))
  await assert.rejects(resource.promise, { name: 'AbortError' })
})

test('neighbor preloading requests dimensions without decoding images and ignores stale results', async () => {
  let finish
  const context = {
    useEffectEvent: fn => fn,
    currentSessionId: 'session', imageResourcesRef: ref({}), pendingLoadsRef: ref({}),
    sessionVersionRef: ref(0), imageIdSetRef: ref(new Set(['image'])), retainedImageIdsRef: ref(new Set(['image'])),
    commitImageResources(update) { context.imageResourcesRef.current = update(context.imageResourcesRef.current) },
    fetchLocalImageMetadata: () => new Promise(resolve => { finish = resolve }),
    preloadImage() { throw Error('neighbor must not decode the original image') },
    isImagePreloadAbort: error => error.name === 'AbortError',
  }
  const ensure = appHandler('ensureImageResource', context)
  const first = ensure({ id: 'image', url: '/image' })
  finish({ width: 8000, height: 6000 })
  await first
  assert.deepEqual(context.imageResourcesRef.current.image, { width: 8000, height: 6000, status: 'ready' })
  context.imageResourcesRef.current = {}
  const stale = ensure({ id: 'image', url: '/image' })
  context.retainedImageIdsRef.current.clear()
  context.imageResourcesRef.current = {}
  finish({ width: 8000, height: 6000 })
  await stale
  assert.deepEqual(context.imageResourcesRef.current, {})
})

test('cancelled annotation loads cannot repopulate the cache or overwrite a newer load state', async () => {
  let finish
  const context = {
    useEffectEvent: fn => fn, currentSessionId: 'session',
    pendingAnnotationLoadsRef: ref({}), annotationLoadControllersRef: ref({}), annotationLoadStateRef: ref({}),
    sessionVersionRef: ref(0), imageIdSetRef: ref(new Set(['image'])), annotationsByImageRef: ref({}),
    projectClasses: ['cat'],
    setAnnotationLoadState(id, state) { context.annotationLoadStateRef.current[id] = state },
    setAnnotationLoadStates() {},
    fetchLocalAnnotations: () => new Promise(resolve => { finish = resolve }),
    buildDisplayAnnotations() { throw Error('cancelled payload must not be consumed') },
  }
  const ensure = appHandler('ensureAnnotationsLoaded', context)
  const cancel = appHandler('cancelPendingAnnotationLoads', context)
  const pending = ensure({ id: 'image' })
  const controller = context.annotationLoadControllersRef.current.image
  cancel(['image'])
  assert.equal(controller.signal.aborted, true)
  context.annotationLoadStateRef.current.image = 'ready'
  finish({ annotations: [{ label: 'old' }] })
  await pending
  assert.equal(context.annotationLoadStateRef.current.image, 'ready')
  assert.deepEqual(context.annotationsByImageRef.current, {})
})

test('API cancellation stays connected while a large annotation response body downloads', async () => {
  const { fetchLocalAnnotations } = await loadLibrary('api')
  const originalFetch = globalThis.fetch
  let readingBody
  const bodyStarted = new Promise(resolve => { readingBody = resolve })
  globalThis.fetch = async (_url, { signal }) => ({
    ok: true,
    json() {
      readingBody()
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('cancelled', 'AbortError')), { once: true }))
    },
  })
  try {
    const controller = new AbortController()
    const pending = fetchLocalAnnotations('session', 'image', controller.signal)
    await bodyStarted
    controller.abort()
    await assert.rejects(pending, { name: 'AbortError' })
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('restore errors remain visible after its own loading state causes an effect rerun', async () => {
  let rejectRequest
  const request = new Promise((_resolve, reject) => { rejectRequest = reject })
  const context = {
    backendStatus: 'online', restoreAttemptedRef: ref(false), restoreControllerRef: ref(null),
    persistedSessionState: { sourceKind: 'image', sourcePath: '/missing.jpg' },
    hasSession: false, isOpeningSession: false, isMountedRef: ref(true), error: null,
    setSessionError(error) { context.error = error },
    setOpeningTarget() {}, setSessionLoadProgress() {},
    setIsOpeningSession(value) { context.isOpeningSession = value },
    openLocalImagePath: () => request,
    commitPersistedSessionState(value) { context.persistedSessionState = value },
  }
  const restore = appEffect('const restoreSession = async () =>', context)
  assert.equal(restore(), undefined) // Changing isOpeningSession must not cancel this request.
  assert.equal(context.isOpeningSession, true)
  restore()
  rejectRequest(Error('dataset cannot be read'))
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(context.error, 'dataset cannot be read')
  assert.equal(context.isOpeningSession, false)
  assert.equal(context.restoreControllerRef.current, null)
})

test('mount state is restored after a StrictMode cleanup so loading can finish', () => {
  const context = {
    isMountedRef: ref(true), restoreControllerRef: ref(null), sessionVersionRef: ref(0),
    currentSessionIdRef: ref('session'), imageIdSetRef: ref(new Set()),
    pendingAnnotationLoadsRef: ref({}), annotationLoadStateRef: ref({}),
    cancelPendingImageLoads() {}, cancelPendingAnnotationLoads() {}, clearScheduledAnnotationSave() {},
  }
  const mount = appEffect('restoreControllerRef.current?.abort()', context)
  mount()()
  assert.equal(context.isMountedRef.current, false)
  mount()
  assert.equal(context.isMountedRef.current, true)
})
