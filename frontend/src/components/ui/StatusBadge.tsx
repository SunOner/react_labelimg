import type { ReactNode } from 'react'

type StatusBadgeTone = 'default' | 'online' | 'offline'

type StatusBadgeProps = {
  tone?: StatusBadgeTone
  children: ReactNode
  className?: string
}

export function StatusBadge({
  tone = 'default',
  children,
  className,
}: StatusBadgeProps) {
  const toneClass =
    tone === 'online'
      ? 'status-pill is-online'
      : tone === 'offline'
        ? 'status-pill is-offline'
        : 'status-pill'

  return <span className={[toneClass, className].filter(Boolean).join(' ')}>{children}</span>
}
