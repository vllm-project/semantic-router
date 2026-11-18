import { AnimatePresence, motion } from 'framer-motion'
import { Moon, Sun } from 'lucide-react'
import { Button, Tooltip } from '../ui'
import { useTheme } from './theme'

export function ThemeToggle() {
  const { resolved, toggle } = useTheme()
  return (
    <Tooltip content={resolved === 'dark' ? 'Light mode' : 'Dark mode'}>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label="Toggle theme"
        onClick={(event) => toggle({ x: event.clientX, y: event.clientY })}
        className="overflow-hidden"
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={resolved}
            initial={{ y: -14, opacity: 0, rotate: -40 }}
            animate={{ y: 0, opacity: 1, rotate: 0 }}
            exit={{ y: 14, opacity: 0, rotate: 40 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="grid place-items-center"
          >
            {resolved === 'dark' ? <Moon /> : <Sun />}
          </motion.span>
        </AnimatePresence>
      </Button>
    </Tooltip>
  )
}
