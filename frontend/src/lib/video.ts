export function formatVideoTime(seconds: number) {
  const milliseconds = Math.round(Math.max(0, seconds) * 1000)
  const hours = Math.floor(milliseconds / 3_600_000)
  const minutes = Math.floor(milliseconds / 60_000) % 60
  const secs = Math.floor(milliseconds / 1000) % 60
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(milliseconds % 1000).padStart(3, '0')}`
}

export function videoFrameKey(seconds: number) {
  return Math.round(seconds * 1_000_000).toString()
}

export function videoFrameFilename(
  name: string,
  sourceId: string,
  seconds: number,
  format: 'png' | 'jpg',
) {
  const stem =
    name
      .replace(/\.[^.]+$/, '')
      .replace(/[<>:"/\\|?*]/g, '_')
      .split('')
      .map((character) => (character.charCodeAt(0) < 32 ? '_' : character))
      .join('')
      .slice(0, 90) || 'video'
  return `${stem}_${sourceId}_t${videoFrameKey(seconds).padStart(12, '0')}us.${format}`
}

export function boundedVideoTime(seconds: number, duration: number) {
  // Seeking to duration can yield an ended/blank frame.
  return Math.max(0, Math.min(seconds, Math.max(0, duration - 0.000001)))
}

export async function identifyVideo(file: File) {
  // Bounded sampling avoids loading multi-gigabyte videos into memory.
  const sample = new Blob([
    file.name,
    String(file.size),
    file.slice(0, 65536),
    file.slice(Math.max(0, file.size - 65536)),
  ])
  const bytes = new Uint8Array(await sample.arrayBuffer())
  let hash = 0xcbf29ce484222325n
  for (const byte of bytes)
    hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n)
  return hash.toString(16).padStart(16, '0')
}

export function captureVideoFrame(
  video: HTMLVideoElement,
  format: 'png' | 'jpg',
): Promise<Blob> {
  if (
    video.seeking ||
    video.readyState < 2 ||
    !video.videoWidth ||
    !video.videoHeight
  ) {
    return Promise.reject(
      new Error('Wait for the video frame to finish loading.'),
    )
  }
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const context = canvas.getContext('2d')
  if (!context)
    return Promise.reject(new Error('Could not create a frame image.'))
  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        canvas.width = 0
        canvas.height = 0
        if (blob) resolve(blob)
        else reject(new Error('Could not encode the video frame.'))
      },
      format === 'png' ? 'image/png' : 'image/jpeg',
      0.95,
    )
  })
}
