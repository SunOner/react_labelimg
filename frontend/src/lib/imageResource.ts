// A single displayed image owns its object URL. Navigating away cancels the
// download and revokes the URL, releasing the browser's decoded image resource.
export function createImageResource(
  url: string,
  io = { fetch: globalThis.fetch.bind(globalThis), create: URL.createObjectURL.bind(URL), revoke: URL.revokeObjectURL.bind(URL) },
) {
  const controller = new AbortController()
  let objectUrl: string | null = null
  let disposed = false
  const promise = io.fetch(url, { signal: controller.signal })
    .then(async response => {
      if (!response.ok) throw new Error(`Failed to load image (${response.status})`)
      const blob = await response.blob()
      if (disposed) throw new DOMException('Image load cancelled', 'AbortError')
      objectUrl = io.create(blob)
      return objectUrl
    })
  const dispose = () => {
    disposed = true
    controller.abort()
    if (objectUrl !== null) { io.revoke(objectUrl); objectUrl = null }
  }
  return { promise, dispose, isDisposed: () => disposed }
}
