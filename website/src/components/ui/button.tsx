import * as React from 'react'
import { twMerge } from 'tailwind-merge'

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' }
/** Accessible shadcn/ui-style button primitive with project-specific variants. */
export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', type = 'button', ...props }, ref) => <button ref={ref} type={type} className={twMerge('btn', variant, className)} {...props}/>
)
Button.displayName = 'Button'
