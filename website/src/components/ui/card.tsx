import * as React from 'react'
import { twMerge } from 'tailwind-merge'

/** Local shadcn/ui-style primitive, themed for the research interface. */
export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={twMerge('card', className)} {...props}/>
)
Card.displayName = 'Card'
