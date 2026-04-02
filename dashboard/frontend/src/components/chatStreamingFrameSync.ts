const requestFrame = (callback: () => void): number => {
  if (typeof globalThis.requestAnimationFrame === 'function') {
    return globalThis.requestAnimationFrame(callback)
  }

  return globalThis.setTimeout(callback, 16)
}

const cancelFrame = (handle: number) => {
  if (typeof globalThis.cancelAnimationFrame === 'function') {
    globalThis.cancelAnimationFrame(handle)
    return
  }

  globalThis.clearTimeout(handle)
}

export const createFrameSyncController = (callback: () => void) => {
  let handle: number | null = null

  return {
    schedule() {
      if (handle !== null) {
        return
      }

      handle = requestFrame(() => {
        handle = null
        callback()
      })
    },
    drain() {
      if (handle === null) {
        return
      }

      cancelFrame(handle)
      handle = null
      callback()
    },
    flush() {
      if (handle !== null) {
        cancelFrame(handle)
        handle = null
      }
      callback()
    },
    cancel() {
      if (handle === null) {
        return
      }

      cancelFrame(handle)
      handle = null
    },
  }
}
