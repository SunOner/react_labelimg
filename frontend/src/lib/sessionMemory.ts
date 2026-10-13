import type { Annotation } from '../types'

export const MAX_CACHED_ANNOTATION_IMAGES = 24
export const MAX_CACHED_ANNOTATIONS = 20_000

// Current/unsaved/SAM images must survive eviction. Everything else is a
// disposable read cache; revisiting an image loads its annotations from disk.
export function trimAnnotationCache<T extends { length: number }>(
  cache: Record<string, T>, preferredIds: Iterable<string>, protectedIds: Set<string>,
  maxImages = MAX_CACHED_ANNOTATION_IMAGES, maxAnnotations = MAX_CACHED_ANNOTATIONS,
) {
  const keep = new Set<string>()
  let count = 0
  for (const id of protectedIds) {
    if (cache[id] !== undefined) {
      keep.add(id)
      count += cache[id].length
    }
  }
  const candidates = [...preferredIds, ...Object.keys(cache).reverse()]
  for (const id of candidates) {
    if (keep.has(id) || cache[id] === undefined) continue
    if (keep.size >= maxImages || count + cache[id].length > maxAnnotations) continue
    keep.add(id)
    count += cache[id].length
  }
  if (keep.size === Object.keys(cache).length) return cache
  return Object.fromEntries(Object.entries(cache).filter(([id]) => keep.has(id)))
}

export type AnnotationSummary = {
  label: string
  sourceClassIndex?: number | null
  hasUnknownClass?: boolean
  count: number
  difficultCount: number
}

export function summarizeAnnotations(annotations: Annotation[]): AnnotationSummary[] {
  const groups = new Map<string, AnnotationSummary>()
  for (const annotation of annotations) {
    const key = JSON.stringify([annotation.label, annotation.sourceClassIndex, !!annotation.hasUnknownClass])
    let group = groups.get(key)
    if (!group) {
      group = { label: annotation.label, sourceClassIndex: annotation.sourceClassIndex,
        hasUnknownClass: annotation.hasUnknownClass, count: 0, difficultCount: 0 }
      groups.set(key, group)
    }
    group.count++
    if (annotation.difficult) group.difficultCount++
  }
  return [...groups.values()]
}

export function aggregateAnnotationSummaries(
  summaries: Iterable<AnnotationSummary[]>,
  resolveLabel: (group: AnnotationSummary) => string,
  isUsable: (group: AnnotationSummary) => boolean,
) {
  const classCounts: Record<string, number> = {}
  const imageCounts: Record<string, number> = {}
  let difficultCount = 0
  let unknownCount = 0
  for (const groups of summaries) {
    const imageLabels = new Set<string>()
    for (const group of groups) {
      if (!isUsable(group)) { unknownCount += group.count; continue }
      const label = resolveLabel(group)
      classCounts[label] = (classCounts[label] ?? 0) + group.count
      difficultCount += group.difficultCount
      imageLabels.add(label)
    }
    for (const label of imageLabels) imageCounts[label] = (imageCounts[label] ?? 0) + 1
  }
  return { classCounts, imageCounts, difficultCount, unknownCount }
}

export function renameAnnotationSummaries(summaries: Map<string, AnnotationSummary[]>, source: string, target: string) {
  for (const [id, groups] of summaries) {
    summaries.set(id, groups.map(group => group.label === source ? { ...group, label: target } : group))
  }
}
