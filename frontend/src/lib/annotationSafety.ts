export type AnnotationLoadState = 'idle' | 'loading' | 'ready' | 'error'

export function canEditAnnotations(state: AnnotationLoadState | undefined) {
  return state === 'ready'
}

export function renameProjectClass(classes: string[], source: string, target: string) {
  if (classes.includes(source)) {
    return classes.map((label) => label === source ? target : label)
  }
  return classes.includes(target) ? [...classes] : [...classes, target]
}

export function resolveProjectClassLabel(
  annotation: { label: string; sourceClassIndex?: number | null; hasUnknownClass?: boolean },
  classes: string[],
) {
  // A known label comes from classes.txt/VOC. Project aliases may only resolve
  // unknown numeric classes; they must never reinterpret a valid disk label.
  if (!annotation.hasUnknownClass) {
    return annotation.label.trim() || 'object'
  }
  const index = annotation.sourceClassIndex
  if (typeof index === 'number' && Number.isInteger(index) && index >= 0 && classes[index]) {
    return classes[index]
  }
  return annotation.label.trim() || 'object'
}

type Ref<T> = { current: T }
type QueuedSave = { imageId: string; revision: number }

export async function flushQueuedAnnotationSave<T extends QueuedSave>(
  imageId: string,
  options: {
    timeouts: Ref<Record<string, number>>
    queued: Ref<Record<string, T>>
    running: Ref<Record<string, Promise<void>>>
    clearTimeout: (id: number) => void
    runSave: (save: T) => Promise<void>
    isActive: (save: T) => boolean
  },
): Promise<void> {
  const { timeouts, queued, running } = options
  const timeoutId = timeouts.current[imageId]
  if (timeoutId !== undefined) {
    options.clearTimeout(timeoutId)
    delete timeouts.current[imageId]
  }
  const previousSave = running.current[imageId]
  if (previousSave) {
    try {
      await previousSave
    } catch (error) {
      if (!queued.current[imageId]) throw error
    }
    // Another flush may already have consumed the next revision while we
    // awaited this request. Recheck both the queue and the running promise.
    return flushQueuedAnnotationSave(imageId, options)
  }
  const save = queued.current[imageId]
  if (!save) return
  delete queued.current[imageId]
  const promise = Promise.resolve().then(() => options.runSave(save))
  running.current[imageId] = promise
  try {
    await promise
  } catch (error) {
    const newerSave = queued.current[imageId]
    if (options.isActive(save) && (!newerSave || newerSave.revision <= save.revision)) {
      queued.current[imageId] = save
    }
    throw error
  } finally {
    if (running.current[imageId] === promise) delete running.current[imageId]
  }
  if (queued.current[imageId]) await flushQueuedAnnotationSave(imageId, options)
}
