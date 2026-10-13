import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
import { appHandler } from './appHandler.mjs'
const safetySource = fs.readFileSync(new URL('../src/lib/annotationSafety.ts', import.meta.url), 'utf8')
const safetyModule = ts.transpileModule(safetySource, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText
const {
  canEditAnnotations,
  flushQueuedAnnotationSave,
  renameProjectClass,
  resolveProjectClassLabel,
} = await import(`data:text/javascript,${encodeURIComponent(safetyModule)}`)

function queue(runSave) {
  return {
    timeouts: { current: {} },
    queued: { current: { image: { imageId: 'image', revision: 1 } } },
    running: { current: {} },
    clearTimeout() {},
    runSave,
    isActive: () => true,
  }
}

test('failed autosave remains pending and a retry persists it', async () => {
  let attempts = 0
  const options = queue(async () => { if (++attempts === 1) throw Error('disk full') })
  await assert.rejects(flushQueuedAnnotationSave('image', options), /disk full/)
  assert.equal(options.queued.current.image.revision, 1)
  assert.deepEqual(options.running.current, {})
  await flushQueuedAnnotationSave('image', options)
  assert.equal(attempts, 2)
  assert.deepEqual(options.queued.current, {})
})

test('failed older save cannot replace newer edits in the queue', async () => {
  const persisted = []
  const options = queue(async save => {
    if (save.revision === 1) {
      options.queued.current.image = { imageId: 'image', revision: 2 }
      throw Error('network failure')
    }
    persisted.push(save.revision)
  })
  await assert.rejects(flushQueuedAnnotationSave('image', options))
  assert.equal(options.queued.current.image.revision, 2)
  await flushQueuedAnnotationSave('image', options)
  assert.deepEqual(persisted, [2])
})

test('concurrent flushes serialize writes for one image', async () => {
  let finish
  let active = 0
  let maximum = 0
  const persisted = []
  const options = queue(async save => {
    maximum = Math.max(maximum, ++active)
    if (save.revision === 1) await new Promise(resolve => { finish = resolve })
    persisted.push(save.revision)
    active--
  })
  const first = flushQueuedAnnotationSave('image', options)
  await Promise.resolve()
  options.queued.current.image = { imageId: 'image', revision: 2 }
  const second = flushQueuedAnnotationSave('image', options)
  finish()
  await Promise.all([first, second])
  assert.deepEqual(persisted, [1, 2])
  assert.equal(maximum, 1)
})

test('an obsolete session does not receive a failed save', async () => {
  const options = queue(async () => { throw Error('session removed') })
  options.isActive = () => false
  await assert.rejects(flushQueuedAnnotationSave('image', options))
  assert.deepEqual(options.queued.current, {})
})

test('a session-change flush waits for a newer save already consumed by another flush', async () => {
  const finishes = {}
  const options = queue(save => new Promise(resolve => { finishes[save.revision] = resolve }))
  const first = flushQueuedAnnotationSave('image', options)
  await Promise.resolve()
  options.queued.current.image = { imageId: 'image', revision: 2 }
  let sessionFlushFinished = false
  const sessionFlush = flushQueuedAnnotationSave('image', options).then(() => { sessionFlushFinished = true })
  finishes[1]()
  for (let tick = 0; tick < 6; tick++) await Promise.resolve()
  assert.equal(typeof finishes[2], 'function')
  assert.equal(sessionFlushFinished, false)
  finishes[2]()
  await Promise.all([first, sessionFlush])
  assert.equal(sessionFlushFinished, true)
})

test('renaming retains class positions and known disk labels', () => {
  const classes = renameProjectClass(['cat', 'dog'], 'cat', 'kitten')
  assert.deepEqual(classes, ['kitten', 'dog'])
  assert.equal(resolveProjectClassLabel({ label: 'dog', sourceClassIndex: 1 }, classes), 'dog')
  assert.equal(resolveProjectClassLabel({ label: 'kitten', sourceClassIndex: 0 }, classes), 'kitten')
  assert.equal(resolveProjectClassLabel({ label: 'cat', sourceClassIndex: 0 }, ['dog', 'cat']), 'cat')
  assert.equal(resolveProjectClassLabel({ label: 'class_0', sourceClassIndex: 0, hasUnknownClass: true }, classes), 'kitten')
})

test('loading and failed annotations cannot be edited or saved', () => {
  for (const state of [undefined, 'idle', 'loading', 'error']) {
    assert.equal(canEditAnnotations(state), false)
  }
  assert.equal(canEditAnnotations('ready'), true)
})

// Run the actual App handlers with a failed API request. This checks that the
// load state gate is wired into the drawing path, not just correct in isolation.
test('failed annotation load blocks drawing and explicit retry preserves existing boxes', async () => {
  const loaded = [{ id: 'existing', label: 'cat', x: 5, y: 5, width: 10, height: 10 }]
  const context = {
    useEffectEvent: fn => fn,
    currentSessionId: 'session',
    currentEntry: { id: 'image' },
    annotationLoadStateRef: { current: {} },
    pendingAnnotationLoadsRef: { current: {} },
    annotationLoadControllersRef: { current: {} },
    sessionVersionRef: { current: 0 },
    imageIdSetRef: { current: new Set(['image']) },
    annotationsByImageRef: { current: {} },
    projectClasses: ['cat'],
    errors: {},
    attempts: 0,
    isRenamingClass: false,
    canEditAnnotations,
    setAnnotationLoadState(id, state) { context.annotationLoadStateRef.current[id] = state },
    setAnnotationLoadErrors(update) { context.errors = update(context.errors) },
    async fetchLocalAnnotations() {
      if (++context.attempts === 1) throw Error('temporary outage')
      return { annotations: loaded, count: 1 }
    },
    buildDisplayAnnotations: value => value,
    setImages() {},
    countLabelingEligibleAnnotations: value => value.length,
    commitAnnotationsByImage(update) { context.annotationsByImageRef.current = update(context.annotationsByImageRef.current) },
    applyAutomaticAnnotationDeduplication() {},
    setSelectedId() {},
    updateDrawStart() { throw Error('drawing must be blocked') },
    setDraftRect() {},
  }
  const ensure = appHandler('ensureAnnotationsLoaded', context)
  const draw = appHandler('handleCanvasPointerDown', context)
  await ensure({ id: 'image' })
  assert.equal(context.annotationLoadStateRef.current.image, 'error')
  assert.match(context.errors.image, /temporary outage/)
  draw({ x: 20, y: 20 })
  await ensure({ id: 'image' })
  assert.equal(context.attempts, 1) // No uncontrolled retry loop.
  await ensure({ id: 'image' }, true)
  assert.equal(context.annotationLoadStateRef.current.image, 'ready')
  assert.deepEqual(context.annotationsByImageRef.current.image, loaded)
  assert.deepEqual(context.errors, {})
})
