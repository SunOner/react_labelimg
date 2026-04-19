import type { Annotation, LoadedImage, Point, Rect } from '../types'

const COLOR_PALETTE = [
  '#db893e',
  '#c04d2f',
  '#355f8c',
  '#2f7d6b',
  '#8f5f99',
  '#65743a',
  '#b35c63',
  '#505f7b',
]

export const MIN_BOX_SIZE = 10

export function rectFromPoints(start: Point, end: Point): Rect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  }
}

export function buildClassList(annotations: Annotation[]) {
  const labels = new Set<string>()

  for (const annotation of annotations) {
    const label = annotation.label.trim()
    if (label) {
      labels.add(label)
    }
  }

  return [...labels]
}

export function buildYoloClassList(
  annotations: Annotation[],
  preferredClasses: string[] = [],
) {
  const labels = new Set<string>()

  for (const preferredClass of preferredClasses) {
    const normalized = preferredClass.trim()
    if (normalized) {
      labels.add(normalized)
    }
  }

  for (const annotation of annotations) {
    const normalized = annotation.label.trim()
    if (normalized) {
      labels.add(normalized)
    }
  }

  return [...labels]
}

export function labelToColor(label: string) {
  const normalized = label.trim() || 'object'
  let hash = 0

  for (const char of normalized) {
    hash = (hash << 5) - hash + char.charCodeAt(0)
    hash |= 0
  }

  return COLOR_PALETTE[Math.abs(hash) % COLOR_PALETTE.length]
}

export function downloadTextFile(
  filename: string,
  contents: string,
  mimeType = 'text/plain;charset=utf-8',
) {
  const blob = new Blob([contents], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function serializeSession(
  image: LoadedImage,
  annotations: Annotation[],
) {
  return JSON.stringify(
    {
      version: 1,
      createdAt: new Date().toISOString(),
      image: {
        name: image.name,
        width: image.width,
        height: image.height,
      },
      classes: buildClassList(annotations),
      annotations: annotations.map((annotation) => ({
        id: annotation.id,
        label: annotation.label.trim() || 'object',
        difficult: annotation.difficult,
        color: annotation.color,
        x: round(annotation.x),
        y: round(annotation.y),
        width: round(annotation.width),
        height: round(annotation.height),
      })),
    },
    null,
    2,
  )
}

export function serializePascalVoc(
  image: LoadedImage,
  annotations: Annotation[],
) {
  const objects = annotations
    .map((annotation) => {
      const box = annotationToBoundingBox(annotation, image)
      const truncated =
        box.xMin === 1 ||
        box.yMin === 1 ||
        box.xMax === image.width ||
        box.yMax === image.height

      return [
        '\t<object>',
        `\t\t<name>${escapeXml(annotation.label.trim() || 'object')}</name>`,
        '\t\t<pose>Unspecified</pose>',
        `\t\t<truncated>${truncated ? 1 : 0}</truncated>`,
        `\t\t<difficult>${annotation.difficult ? 1 : 0}</difficult>`,
        '\t\t<bndbox>',
        `\t\t\t<xmin>${box.xMin}</xmin>`,
        `\t\t\t<ymin>${box.yMin}</ymin>`,
        `\t\t\t<xmax>${box.xMax}</xmax>`,
        `\t\t\t<ymax>${box.yMax}</ymax>`,
        '\t\t</bndbox>',
        '\t</object>',
      ].join('\n')
    })
    .join('\n')

  return [
    '<?xml version="1.0" encoding="utf-8"?>',
    '<annotation verified="no">',
    '\t<folder>browser</folder>',
    `\t<filename>${escapeXml(image.name)}</filename>`,
    '\t<source>',
    '\t\t<database>Unknown</database>',
    '\t</source>',
    '\t<size>',
    `\t\t<width>${image.width}</width>`,
    `\t\t<height>${image.height}</height>`,
    '\t\t<depth>3</depth>',
    '\t</size>',
    '\t<segmented>0</segmented>',
    objects,
    '</annotation>',
  ]
    .filter(Boolean)
    .join('\n')
}

export function serializeYolo(
  image: LoadedImage,
  annotations: Annotation[],
  preferredClasses: string[] = [],
) {
  const classes = buildYoloClassList(annotations, preferredClasses)

  const annotationText = annotations
    .map((annotation) => {
      const label = annotation.label.trim() || 'object'
      const classIndex = classes.indexOf(label)
      const xCenter = (annotation.x + annotation.width / 2) / image.width
      const yCenter = (annotation.y + annotation.height / 2) / image.height
      const width = annotation.width / image.width
      const height = annotation.height / image.height

      return [
        classIndex,
        xCenter.toFixed(6),
        yCenter.toFixed(6),
        width.toFixed(6),
        height.toFixed(6),
      ].join(' ')
    })
    .join('\n')

  return {
    annotationText,
    classesText: classes.join('\n'),
  }
}

function annotationToBoundingBox(annotation: Annotation, image: LoadedImage) {
  const xMin = clamp(Math.round(annotation.x), 1, image.width)
  const yMin = clamp(Math.round(annotation.y), 1, image.height)
  const xMax = clamp(
    Math.round(annotation.x + annotation.width),
    1,
    image.width,
  )
  const yMax = clamp(
    Math.round(annotation.y + annotation.height),
    1,
    image.height,
  )

  return { xMin, yMin, xMax, yMax }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function escapeXml(value: string) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function round(value: number) {
  return Math.round(value * 100) / 100
}
