import { AppButton } from './AppButton'

type Props = {
  canUndo: boolean
  canRedo: boolean
  disabled?: boolean
  onUndo: () => void
  onRedo: () => void
  undoShortcut?: string
  redoShortcut?: string
}

export function HistoryButtons({ canUndo, canRedo, disabled, onUndo, onRedo,
  undoShortcut = 'Ctrl+Z', redoShortcut = 'Ctrl+Y / Ctrl+Shift+Z' }: Props) {
  return (
    <div className="annotation-history-controls" role="group" aria-label="Annotation history">
      <AppButton disabled={disabled || !canUndo} onClick={onUndo} aria-label="Undo" title={undoShortcut ? `Undo (${undoShortcut})` : 'Undo'}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M9 4 4 9l5 5M4 9h10a6 6 0 0 1 0 12h-3" /></svg>
      </AppButton>
      <AppButton disabled={disabled || !canRedo} onClick={onRedo} aria-label="Redo" title={redoShortcut ? `Redo (${redoShortcut})` : 'Redo'}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="m15 4 5 5-5 5m5-5H10a6 6 0 0 0 0 12h3" /></svg>
      </AppButton>
    </div>
  )
}
