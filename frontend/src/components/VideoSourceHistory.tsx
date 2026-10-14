import { useEffect, useRef, useState } from 'react'
import { AppButton } from './ui/AppButton'
import { formatVideoTime } from '../lib/video'
import type { VideoMetadata } from '../lib/api'

type Props = {
  videos: VideoMetadata[]
  error: string | null
  busy: boolean
  currentId?: string
  currentFrameCount: number
  onRefresh: () => void
  onChooseBrowserFile: () => void
  onChooseOriginalFile: () => void
  onOpenPath: (path: string) => void
}

export function VideoSourceHistory({
  videos,
  error,
  busy,
  currentId,
  currentFrameCount,
  onRefresh,
  onChooseBrowserFile,
  onChooseOriginalFile,
  onOpenPath,
}: Props) {
  const [path, setPath] = useState('')
  const panelRef = useRef<HTMLDetailsElement>(null)
  const close = () => {
    if (panelRef.current) panelRef.current.open = false
  }
  useEffect(() => {
    const dismiss = (event: PointerEvent) => {
      const panel = panelRef.current
      if (panel && !panel.contains(event.target as Node)) panel.open = false
    }
    document.addEventListener('pointerdown', dismiss)
    return () => document.removeEventListener('pointerdown', dismiss)
  }, [])
  return (
    <details className="video-source-history" ref={panelRef} onKeyDown={(event) => {
      // Keep transport and annotation shortcuts out of the source picker.
      if (event.key !== 'Tab' && (event.key !== 'Escape' || panelRef.current?.open)) event.stopPropagation()
      if (event.key === 'Escape' && panelRef.current?.open) {
        event.preventDefault()
        close()
        panelRef.current.querySelector('summary')?.focus()
      }
    }}>
      <summary>
        <span>Sources</span>
        <span className="video-source-count">{videos.length}</span>
        {error ? <span className="video-source-warning" title={error} aria-label="Video history unavailable">!</span> : null}
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
      </summary>
      <div className="video-source-panel">
        <div className="video-source-panel-heading">
          <strong>Recent videos</strong>
          <AppButton disabled={busy} onClick={onRefresh} aria-label="Refresh video history" title="Refresh history">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5M5 8a7 7 0 0 1 12-3l3 4M19 16a7 7 0 0 1-12 3l-3-4" /></svg>
          </AppButton>
        </div>
        {error ? (
          <p className="video-error" role="alert">
            Could not load video history: {error}
          </p>
        ) : null}
        <div className="video-history-list">
          {videos.map((video) => (
            <div key={video.id} className="video-history-row" aria-current={video.id === currentId ? 'true' : undefined}>
              <div>
                <strong>{video.name}</strong>
                <span title={video.sourcePath ?? undefined}>
                  {video.sourcePath ?? 'Select the original file to resume'}
                </span>
                <span>
                  {formatVideoTime(video.state.positionSeconds)} ·{' '}
                  {video.id === currentId ? currentFrameCount : video.frameCount}{' '}
                  frames
                  {video.sourceMissing
                    ? ' · File unavailable'
                    : ''}
                </span>
              </div>
              <AppButton
                disabled={busy}
                onClick={() => {
                  close()
                  return video.sourcePath && !video.sourceMissing
                    ? onOpenPath(video.sourcePath)
                    : onChooseBrowserFile()
                }}
              >
                {video.sourcePath && !video.sourceMissing
                  ? 'Resume'
                  : 'Select file'}
              </AppButton>
            </div>
          ))}
          {!videos.length ? (
            <p className="muted">
              No recent videos in this dataset.
            </p>
          ) : null}
        </div>
        <div className="video-source-open">
          <AppButton disabled={busy} onClick={() => { close(); onChooseOriginalFile() }}>
            Browse original file
          </AppButton>
          <details className="video-source-path">
            <summary>Open by path</summary>
            <div className="video-source-controls">
              <label>
                Video path
                <input value={path} onChange={(event) => setPath(event.target.value)}
                  placeholder="C:\Videos\clip.mp4 or /media/clip.mp4" disabled={busy}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && path.trim() && !busy) {
                      event.preventDefault()
                      close()
                      onOpenPath(path.trim())
                    }
                  }} />
              </label>
              <AppButton disabled={busy || !path.trim()} onClick={() => { close(); onOpenPath(path.trim()) }}>
                Open
              </AppButton>
            </div>
          </details>
        </div>
      </div>
    </details>
  )
}
