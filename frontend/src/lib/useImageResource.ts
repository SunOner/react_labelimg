import { useEffect, useState } from 'react'
import { createImageResource } from './imageResource'

export function useImageResource(source: string | undefined) {
  const [state, setState] = useState<{
    source: string
    url?: string
    error?: boolean
    resource: ReturnType<typeof createImageResource>
  } | null>(null)
  useEffect(() => {
    if (!source) return
    let active = true
    const resource = createImageResource(source)
    resource.promise.then(
      url => { if (active) setState({ source, url, resource }) },
      error => {
        if (active && error?.name !== 'AbortError') setState({ source, error: true, resource })
      },
    )
    return () => { active = false; resource.dispose() }
  }, [source])
  return state && state.source === source && !state.resource.isDisposed() ? state : null
}
