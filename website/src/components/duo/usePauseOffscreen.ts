import { type RefObject, useEffect } from 'react'

// Sets data-offscreen on the element while it is out of view, so its CSS animations can pause there.
export default function usePauseOffscreen(ref: RefObject<HTMLElement>): void {
  useEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) delete element.dataset.offscreen
      else element.dataset.offscreen = ''
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
}
