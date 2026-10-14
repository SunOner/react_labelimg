import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import ts from 'typescript'
import { appHandler } from './appHandler.mjs'

const source = fs.readFileSync(new URL('../src/lib/annotationHistory.ts', import.meta.url), 'utf8')
const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText
const { AnnotationHistory, annotationsEqual } = await import(`data:text/javascript,${encodeURIComponent(code)}`)
const box = (id = 'box', x = 10) => ({ id, label: 'cat', color: '#123456', x, y: 12, width: 30, height: 40, difficult: false })
const snapshot = (annotations, selectedId = annotations[0]?.id ?? null) => ({ annotations, selectedId })

test('undo and redo restore complete boxes, ordering, classes and selection without mutating snapshots', () => {
  const history = new AnnotationHistory()
  const states = [[], [box()], [box(), box('copy', 20)], [box('copy', 22), { ...box(), label: 'dog', difficult: true }], [box('copy', 22)], []]
  for (let i = 1; i < states.length; i++) history.record('image', snapshot(states[i - 1]), snapshot(states[i]))
  states[2][0].x = 999
  for (let i = states.length - 2; i >= 0; i--) {
    const restored = history.undo('image')
    const expected = i === 2 ? [box(), box('copy', 20)] : states[i]
    assert.deepEqual(restored, snapshot(expected))
    if (restored.annotations.length) restored.annotations[0].x = -100
  }
  assert.equal(history.undo('image'), null)
  for (let i = 1; i < states.length; i++) {
    const expected = i === 2 ? [box(), box('copy', 20)] : states[i]
    assert.deepEqual(history.redo('image'), snapshot(expected))
  }
  assert.equal(history.redo('image'), null)
})

test('one drag is one undo step and returning to the start preserves an existing redo branch', () => {
  const history = new AnnotationHistory()
  let current = [box()]
  history.beginGroup('image')
  for (let x = 11; x <= 100; x++) {
    const next = [box('box', x)]
    history.record('image', snapshot(current), snapshot(next))
    current = next
  }
  assert.equal(history.state('image').canUndo, false)
  history.endGroup()
  assert.deepEqual(history.undo('image').annotations, [box()])
  assert.equal(history.undo('image'), null)
  history.beginGroup('image')
  history.record('image', snapshot([box()]), snapshot([box('box', 11)]))
  history.record('image', snapshot([box('box', 11)]), snapshot([box()]))
  history.endGroup()
  assert.equal(history.state('image').canUndo, false)
  assert.deepEqual(history.redo('image').annotations, [box('box', 100)])
})

test('new edits discard redo while no-op edits do not create steps or discard redo', () => {
  const history = new AnnotationHistory()
  history.record('image', snapshot([]), snapshot([box()]))
  history.undo('image')
  assert.equal(history.record('image', snapshot([]), snapshot([], 'different selection')), false)
  assert.equal(history.state('image').canRedo, true)
  history.record('image', snapshot([]), snapshot([box('new')]))
  assert.equal(history.redo('image'), null)
  assert.equal(annotationsEqual([box()], [{ ...box(), hasUnknownClass: false, sourceClassIndex: null }]), true)
})

test('typing a class name is grouped while allowing the user to undo it directly', () => {
  const history = new AnnotationHistory()
  let current = [box()]
  history.beginGroup('frame', false)
  for (const label of ['', 'd', 'do', 'dog']) {
    const next = [{ ...box(), label }]
    history.record('frame', snapshot(current), snapshot(next))
    current = next
  }
  assert.equal(history.state('frame').canUndo, true)
  assert.equal(history.undo('frame').annotations[0].label, 'cat')
  assert.equal(history.redo('frame').annotations[0].label, 'dog')
})

test('histories are isolated per image and bounded by images, steps and annotation count', () => {
  const history = new AnnotationHistory(3, 2, 1000)
  for (const image of ['old', 'first', 'second']) {
    let current = []
    for (let x = 1; x <= 10; x++) {
      const next = [box(image, x)]
      history.record(image, snapshot(current), snapshot(next))
      current = next
    }
  }
  assert.equal(history.undo('old'), null)
  assert.equal(history.undo('first').annotations[0].id, 'first')
  assert.equal(history.undo('second').annotations[0].id, 'second')
  assert.ok(history.undo('second'))
  assert.ok(history.undo('second'))
  assert.equal(history.undo('second'), null)
  const limited = new AnnotationHistory(50, 10, 5)
  const large = Array.from({ length: 100 }, (_, i) => box(String(i)))
  limited.record('old', snapshot([]), snapshot([box()]))
  limited.record('large', snapshot(large), snapshot([]))
  assert.equal(limited.undo('old'), null)
  assert.equal(limited.undo('large').annotations.length, 100)
  limited.clear()
  assert.equal(limited.redo('large'), null)
})

function editorContext() {
  const context = {
    useEffectEvent: value => value, useCallback: value => value, annotationHistory: new AnnotationHistory(),
    currentHistoryKey: 'session:image',
    currentSessionId: 'session', currentEntry: { id: 'image' }, selectedId: null,
    annotationsByImageRef: { current: { image: [] } },
    annotationLoadStateRef: { current: { image: 'ready' } },
    canEditAnnotations: state => state === 'ready', isRenamingClass: false,
    annotationEditingReady: true, isRunningSamAction: false, projectClasses: ['cat'],
    drawStartRef: { current: null }, saved: [], historyState: {}, toasts: [],
    commitAnnotationsByImage(update) { context.annotationsByImageRef.current = update(context.annotationsByImageRef.current) },
    countLabelingEligibleAnnotations: annotations => annotations.length,
    updateImageAnnotationMetadata() {}, setSelectedId(id) { context.selectedId = id },
    scheduleAnnotationSave(id, annotations) { context.saved.push({ id, annotations }) },
    pushToast(message) { context.toasts.push(message) }, buildDisplayAnnotations: annotations => annotations,
    setHistoryAvailability(state) { context.historyState = state }, setOpenMenu() {},
  }
  context.refreshAnnotationHistory = appHandler('refreshAnnotationHistory', context)
  context.replaceImageAnnotations = appHandler('replaceImageAnnotations', context)
  context.restore = appHandler('restoreAnnotationHistory', context)
  return context
}

test('the actual editor undo/redo handlers use autosave without recording their own restoration', () => {
  const context = editorContext()
  context.replaceImageAnnotations('image', [box()], { selectedAnnotationId: 'box' })
  assert.equal(context.historyState.canUndo, true)
  context.restore('undo')
  assert.deepEqual(context.annotationsByImageRef.current.image, [])
  assert.equal(context.selectedId, null)
  assert.equal(context.historyState.canRedo, true)
  context.restore('redo')
  assert.deepEqual(context.annotationsByImageRef.current.image, [box()])
  assert.equal(context.selectedId, 'box')
  assert.deepEqual(context.saved.map(entry => entry.annotations.length), [1, 0, 1])
  assert.equal(context.historyState.canRedo, false)
})

test('history cannot bypass failed annotation loading or a running SAM operation', () => {
  const context = editorContext()
  context.annotationLoadStateRef.current.image = 'error'
  context.replaceImageAnnotations('image', [box()])
  assert.equal(context.saved.length, 0)
  assert.equal(context.historyState.canUndo, undefined)
  context.annotationLoadStateRef.current.image = 'ready'
  context.replaceImageAnnotations('image', [box()])
  context.isRunningSamAction = true
  context.restore('undo')
  assert.equal(context.annotationsByImageRef.current.image.length, 1)
  assert.equal(context.saved.length, 1)
})

test('a button request belonging to a previous image cannot undo the newly selected image', () => {
  const context = editorContext()
  context.replaceImageAnnotations('image', [box()])
  context.restore('undo', 'other-session:image')
  assert.equal(context.annotationsByImageRef.current.image.length, 1)
  assert.equal(context.saved.length, 1)
  assert.equal(context.historyState.canUndo, true)
})

const videoUrl = new URL('../src/components/VideoWorkbench.tsx', import.meta.url)
test('video undo returns to the saved baseline without saving empty frames and redo marks edits dirty', () => {
  const context = {
    annotationHistory: new AnnotationHistory(), annotationsRef: { current: [] },
    baselineAnnotationsRef: { current: [] }, selectedId: null,
    operationRef: { current: false }, dirtyRef: { current: false },
    frame: { image: {} }, seeking: false, drawStartRef: { current: null }, annotationsEqual,
    setAnnotations(annotations) { context.annotations = annotations },
    setSelectedId(id) { context.selectedId = id },
    setDirty(dirty) { context.dirty = dirty },
    setHistoryAvailability(state) { context.historyState = state },
    setMessage() {}, setError() {}, onSave() { assert.fail('Undo must not save a video frame') },
  }
  const edit = appHandler('edit', context, videoUrl)
  const restore = appHandler('restoreAnnotationHistory', context, videoUrl)
  edit([box()])
  assert.equal(context.dirty, true)
  restore('undo')
  assert.deepEqual(context.annotations, [])
  assert.equal(context.dirty, false)
  restore('redo')
  assert.deepEqual(context.annotations, [box()])
  assert.equal(context.dirty, true)
  context.baselineAnnotationsRef.current = [box()]
  edit([])
  restore('undo')
  assert.equal(context.dirty, false)
  assert.equal(context.annotations[0].label, 'cat')
})

test('video Ctrl+Z uses annotation history while text fields keep their native undo', () => {
  const calls = []
  const context = { useEffectEvent: value => value, confirmNavigation: false, busy: false,
    matchesUndoHotkey: undefined, matchesRedoHotkey: undefined,
    restoreAnnotationHistory: direction => calls.push(direction) }
  const key = appHandler('onKeyDown', context, videoUrl)
  const event = { key: 'z', code: 'KeyZ', ctrlKey: true, metaKey: false, shiftKey: false,
    target: { tagName: 'INPUT' }, preventDefault() { calls.push('prevented') } }
  key(event)
  assert.deepEqual(calls, [])
  key({ ...event, target: { tagName: 'BUTTON' } })
  key({ ...event, target: { tagName: 'BUTTON' }, shiftKey: true })
  assert.deepEqual(calls, ['prevented', 'undo', 'prevented', 'redo'])
  context.matchesUndoHotkey = event => event.code === 'KeyU'
  context.matchesRedoHotkey = event => event.code === 'KeyR'
  key({ ...event, target: { tagName: 'BUTTON' } })
  assert.equal(calls.length, 4)
  key({ ...event, code: 'KeyU', target: { tagName: 'BUTTON' } })
  assert.deepEqual(calls.slice(-2), ['prevented', 'undo'])
})
