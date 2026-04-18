export type Point = {
  x: number
  y: number
}

export type Rect = {
  x: number
  y: number
  width: number
  height: number
}

export type Annotation = Rect & {
  id: string
  label: string
  color: string
  difficult: boolean
}

export type ImageEntry = {
  id: string
  name: string
  relativePath: string
  url: string
  annotationCount: number
  annotationFormat?: string | null
}

export type ImageResource = {
  width: number
  height: number
  status: 'idle' | 'loading' | 'ready' | 'error'
}

export type LoadedImage = ImageEntry & {
  width: number
  height: number
}
