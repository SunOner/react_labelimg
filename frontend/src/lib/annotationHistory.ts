import type { Annotation } from '../types'

export type AnnotationSnapshot = { annotations: Annotation[]; selectedId: string | null }
export type HistoryAvailability = { canUndo: boolean; canRedo: boolean }
type Step = { before: AnnotationSnapshot; after: AnnotationSnapshot; group?: number }
type Timeline = { past: Step[]; future: Step[] }

const clone = (snapshot: AnnotationSnapshot): AnnotationSnapshot => ({
  annotations: snapshot.annotations.map((box) => ({ ...box })),
  selectedId: snapshot.selectedId,
})

export function annotationsEqual(left: Annotation[], right: Annotation[]) {
  return left.length === right.length && left.every((box, index) => {
    const other = right[index]
    return box.id === other.id && box.label === other.label && box.color === other.color &&
      box.x === other.x && box.y === other.y && box.width === other.width && box.height === other.height &&
      box.difficult === other.difficult && Boolean(box.hasUnknownClass) === Boolean(other.hasUnknownClass) &&
      (box.sourceClassIndex ?? null) === (other.sourceClassIndex ?? null)
  })
}

// Bound both the number of images and the total retained annotation snapshots.
// A single oversized step for the current image is kept so Undo still works.
export class AnnotationHistory {
  private timelines = new Map<string, Timeline>()
  private sequence = 0
  private group: { key: string; id: number; future: Step[]; blockUndo: boolean } | null = null

  private maxSteps: number
  private maxImages: number
  private maxAnnotations: number

  constructor(maxSteps = 50, maxImages = 10, maxAnnotations = 20_000) {
    this.maxSteps = Math.max(1, maxSteps)
    this.maxImages = Math.max(1, maxImages)
    this.maxAnnotations = Math.max(1, maxAnnotations)
  }

  clear(key?: string) {
    if (key === undefined) { this.timelines.clear(); this.group = null }
    else { this.timelines.delete(key); if (this.group?.key === key) this.group = null }
  }

  beginGroup(key: string, blockUndo = true) {
    this.endGroup()
    this.group = { key, id: ++this.sequence, future: [...(this.timelines.get(key)?.future ?? [])], blockUndo }
  }

  endGroup() { this.group = null }

  state(key: string): HistoryAvailability {
    const timeline = this.timelines.get(key)
    const available = this.group?.key !== key || !this.group.blockUndo
    return { canUndo: available && Boolean(timeline?.past.length), canRedo: available && Boolean(timeline?.future.length) }
  }

  record(key: string, before: AnnotationSnapshot, after: AnnotationSnapshot) {
    if (annotationsEqual(before.annotations, after.annotations)) return false
    const timeline = this.touch(key)
    const group = this.group?.key === key ? this.group : null
    const last = timeline.past.at(-1)
    if (group && last?.group === group.id) {
      last.after = clone(after)
      if (annotationsEqual(last.before.annotations, after.annotations)) {
        timeline.past.pop()
        timeline.future = [...group.future]
      }
    } else {
      timeline.past.push({ before: clone(before), after: clone(after), group: group?.id })
      timeline.future = []
    }
    this.trim(key)
    return true
  }

  undo(key: string) {
    if (!this.state(key).canUndo) return null
    if (this.group?.key === key) this.endGroup()
    const timeline = this.touch(key)
    const step = timeline.past.pop()!
    timeline.future.push(step)
    return clone(step.before)
  }

  redo(key: string) {
    if (!this.state(key).canRedo) return null
    if (this.group?.key === key) this.endGroup()
    const timeline = this.touch(key)
    const step = timeline.future.pop()!
    timeline.past.push(step)
    return clone(step.after)
  }

  private touch(key: string) {
    const timeline = this.timelines.get(key) ?? { past: [], future: [] }
    this.timelines.delete(key)
    this.timelines.set(key, timeline)
    return timeline
  }

  private trim(currentKey: string) {
    for (const timeline of this.timelines.values()) {
      while (timeline.past.length > this.maxSteps) timeline.past.shift()
    }
    while (this.timelines.size > this.maxImages) this.timelines.delete(this.timelines.keys().next().value!)
    const size = () => [...this.timelines.values()].reduce((sum, timeline) =>
      sum + [...timeline.past, ...timeline.future].reduce((count, step) =>
        count + step.before.annotations.length + step.after.annotations.length, 0), 0)
    while (size() > this.maxAnnotations) {
      let removed = false
      for (const [key, timeline] of this.timelines) {
        if (key === currentKey && timeline.past.length + timeline.future.length <= 1) continue
        if (timeline.past.length) timeline.past.shift()
        else if (timeline.future.length) timeline.future.shift()
        else { this.timelines.delete(key); continue }
        removed = true
        break
      }
      if (!removed) break
    }
  }
}
