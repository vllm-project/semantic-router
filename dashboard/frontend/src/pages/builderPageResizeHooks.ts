import { useCallback, useEffect, useRef, useState } from 'react'

interface UseResizableWidthOptions {
  initialWidth: number
  minWidth: number
  getMaxWidth: () => number
  stopPropagation?: boolean
  growthDirection?: 'left' | 'right'
}

function useResizableWidth({
  initialWidth,
  minWidth,
  getMaxWidth,
  stopPropagation = false,
  growthDirection = 'left',
}: UseResizableWidthOptions) {
  const [width, setWidth] = useState(initialWidth)
  const [isDragging, setIsDragging] = useState(false)
  const isDraggingRef = useRef(false)
  const dragStartXRef = useRef(0)
  const dragStartWidthRef = useRef(0)

  const handleDragStart = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault()
      if (stopPropagation) {
        event.stopPropagation()
      }
      isDraggingRef.current = true
      setIsDragging(true)
      dragStartXRef.current = event.clientX
      dragStartWidthRef.current = width
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    },
    [stopPropagation, width],
  )

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!isDraggingRef.current) return
      const pointerDelta = event.clientX - dragStartXRef.current
      const delta = growthDirection === 'right' ? pointerDelta : -pointerDelta
      const maxWidth = Math.max(minWidth, getMaxWidth())
      const nextWidth = Math.min(maxWidth, Math.max(minWidth, dragStartWidthRef.current + delta))
      setWidth(nextWidth)
    }

    const handleDragEnd = () => {
      if (!isDraggingRef.current) return
      isDraggingRef.current = false
      setIsDragging(false)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    document.addEventListener('mousemove', handleDragMove)
    document.addEventListener('mouseup', handleDragEnd)
    return () => {
      document.removeEventListener('mousemove', handleDragMove)
      document.removeEventListener('mouseup', handleDragEnd)
    }
  }, [getMaxWidth, growthDirection, minWidth])

  const adjustWidth = useCallback(
    (delta: number) => {
      setWidth((current) =>
        Math.min(Math.max(minWidth, getMaxWidth()), Math.max(minWidth, current + delta)),
      )
    },
    [getMaxWidth, minWidth],
  )

  return { width, isDragging, handleDragStart, adjustWidth }
}

export { useResizableWidth }
