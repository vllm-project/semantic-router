import { forwardRef, useState, type InputHTMLAttributes } from 'react'
import { Eye, EyeOff, KeyRound } from 'lucide-react'

import { InputGroup, cn } from '../ui'

type PasswordInputProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>

/**
 * Password field with a show/hide toggle. The toggle is named by its content,
 * not `aria-label`, so label lookups for "Password" resolve to the input only.
 */
const PasswordInput = forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ className, ...props }, ref) => {
    const [visible, setVisible] = useState(false)
    return (
      <InputGroup
        ref={ref}
        {...props}
        type={visible ? 'text' : 'password'}
        leading={<KeyRound />}
        className={cn('h-10 pr-10', className)}
        trailing={
          <button
            type="button"
            aria-pressed={visible}
            onClick={() => setVisible((current) => !current)}
            className="grid size-7 place-items-center rounded-md text-subtle-foreground outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            {visible ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            <span className="sr-only">{visible ? 'Hide password' : 'Show password'}</span>
          </button>
        }
      />
    )
  },
)

PasswordInput.displayName = 'PasswordInput'

export default PasswordInput
