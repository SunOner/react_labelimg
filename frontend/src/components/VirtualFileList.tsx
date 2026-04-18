import { useEffect, useRef, useState } from 'react'
import type { ImageEntry } from '../types'

const ROW_HEIGHT = 28
const OVERSCAN = 8

type VirtualFileListProps = {
  images: ImageEntry[]
  currentIndex: number
  onSelectIndex: (index: number) => void
}

export function VirtualFileList({
  images,
  currentIndex,
  onSelectIndex,
}: VirtualFileListProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(320)

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const observer = new ResizeObserver(() => {
      setViewportHeight(container.clientHeight)
    })

    setViewportHeight(container.clientHeight)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const rowTop = currentIndex * ROW_HEIGHT
    const rowBottom = rowTop + ROW_HEIGHT
    const viewTop = container.scrollTop
    const viewBottom = viewTop + container.clientHeight

    if (rowTop < viewTop) {
      container.scrollTop = rowTop
      return
    }

    if (rowBottom > viewBottom) {
      container.scrollTop = rowBottom - container.clientHeight
    }
  }, [currentIndex])

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2
  const endIndex = Math.min(images.length, startIndex + visibleCount)
  const visibleImages = images.slice(startIndex, endIndex)
  const totalHeight = images.length * ROW_HEIGHT

  return (
    <div
      ref={containerRef}
      className="virtual-file-list"
      aria-label="Session images"
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div
        className="virtual-file-list-spacer"
        style={{ height: `${totalHeight}px` }}
      >
        {visibleImages.map((entry, visibleIndex) => {
          const index = startIndex + visibleIndex

          return (
            <button
              key={entry.id}
              type="button"
              title={entry.relativePath}
              className={
                index === currentIndex
                  ? 'file-list-item virtual-file-list-item is-active'
                  : 'file-list-item virtual-file-list-item'
              }
              style={{ transform: `translateY(${index * ROW_HEIGHT}px)` }}
              onClick={() => onSelectIndex(index)}
            >
              <span className="file-list-name">{entry.relativePath}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
