import { useEffect, useRef, useState } from 'react'
import type { ImageEntry } from '../types'

const ROW_HEIGHT = 28
const OVERSCAN = 8

type VirtualFileListEntry = {
  image: ImageEntry
  sourceIndex: number
}

type VirtualFileListProps = {
  entries: VirtualFileListEntry[]
  currentImageId: string | null
  scrollResetKey?: string
  onSelectIndex: (index: number) => void
}

export function VirtualFileList({
  entries,
  currentImageId,
  scrollResetKey,
  onSelectIndex,
}: VirtualFileListProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(320)
  const currentListIndex =
    currentImageId !== null
      ? entries.findIndex((entry) => entry.image.id === currentImageId)
      : -1

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
    if (container) {
      setContainerScrollTop(container, 0, setScrollTop)
    }
  }, [scrollResetKey])

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    if (currentListIndex < 0) {
      return
    }

    const rowTop = currentListIndex * ROW_HEIGHT
    const rowBottom = rowTop + ROW_HEIGHT
    const viewTop = container.scrollTop
    const viewBottom = viewTop + container.clientHeight

    if (rowTop < viewTop) {
      setContainerScrollTop(container, rowTop, setScrollTop)
      return
    }

    if (rowBottom > viewBottom) {
      setContainerScrollTop(
        container,
        rowBottom - container.clientHeight,
        setScrollTop,
      )
    }
  }, [currentListIndex, scrollResetKey, viewportHeight])

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2
  const endIndex = Math.min(entries.length, startIndex + visibleCount)
  const visibleEntries = entries.slice(startIndex, endIndex)
  const totalHeight = entries.length * ROW_HEIGHT

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const maxScrollTop = Math.max(0, totalHeight - container.clientHeight)
    if (container.scrollTop > maxScrollTop) {
      setContainerScrollTop(container, maxScrollTop, setScrollTop)
    }
  }, [totalHeight, viewportHeight])

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
        {visibleEntries.map((entry, visibleIndex) => {
          const index = startIndex + visibleIndex
          const image = entry.image

          return (
            <button
              key={image.id}
              type="button"
              title={image.relativePath}
              className={
                index === currentListIndex
                  ? 'file-list-item virtual-file-list-item is-active'
                  : 'file-list-item virtual-file-list-item'
              }
              style={{ transform: `translateY(${index * ROW_HEIGHT}px)` }}
              onClick={() => onSelectIndex(entry.sourceIndex)}
            >
              <span className="file-list-name">{image.relativePath}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function setContainerScrollTop(
  container: HTMLDivElement,
  nextScrollTop: number,
  setScrollTop: (value: number) => void,
) {
  container.scrollTop = nextScrollTop
  setScrollTop(nextScrollTop)
}
