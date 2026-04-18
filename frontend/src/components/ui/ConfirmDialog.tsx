import { AppButton } from './AppButton'

type ConfirmDialogProps = {
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  confirmTone?: 'danger' | 'default'
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  confirmTone = 'danger',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <div className="lightbox-backdrop" onClick={onCancel}>
      <div
        className="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="confirm-dialog-copy">
          <p className="section-kicker">Confirmation</p>
          <h2>{title}</h2>
          <p className="confirm-dialog-message">{message}</p>
        </div>

        <div className="confirm-dialog-actions">
          <AppButton
            variant="primary"
            className="confirm-dialog-action"
            onClick={onCancel}
          >
            {cancelLabel}
          </AppButton>
          <AppButton
            variant={confirmTone === 'danger' ? 'ghost' : 'primary'}
            className="confirm-dialog-action"
            onClick={onConfirm}
          >
            {confirmLabel}
          </AppButton>
        </div>
      </div>
    </div>
  )
}
