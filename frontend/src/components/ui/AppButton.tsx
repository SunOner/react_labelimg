import type { ButtonHTMLAttributes } from 'react'

type AppButtonVariant =
  | 'menu-trigger'
  | 'menu-item'
  | 'primary'
  | 'chip'
  | 'list-row'
  | 'ghost'
  | 'index'

type AppButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: AppButtonVariant
  isActive?: boolean
}

export function AppButton({
  variant = 'menu-trigger',
  isActive = false,
  className,
  type = 'button',
  ...props
}: AppButtonProps) {
  return (
    <button
      type={type}
      className={joinClasses(
        'ui-button',
        `ui-button--${variant}`,
        isActive ? 'is-active' : '',
        className,
      )}
      {...props}
    />
  )
}

function joinClasses(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}
