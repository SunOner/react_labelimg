import type { ButtonHTMLAttributes } from 'react'
import { AppButton } from './AppButton'

type MenuItemButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  title: string
  description?: string
  shortcut?: string
}

export function MenuItemButton({
  title,
  description,
  shortcut,
  ...props
}: MenuItemButtonProps) {
  return (
    <AppButton variant="menu-item" {...props}>
      <span className="menu-item-copy">
        <strong>{title}</strong>
        {description ? <small>{description}</small> : null}
      </span>
      {shortcut ? <span className="menu-shortcut">{shortcut}</span> : null}
    </AppButton>
  )
}
