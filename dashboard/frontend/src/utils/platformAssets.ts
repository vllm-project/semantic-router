// Preload cache to track which images have been loaded
const preloadedImages = new Set<string>()

// Preload an image and cache it
const preloadImage = (src: string): Promise<void> => {
  if (preloadedImages.has(src)) {
    return Promise.resolve()
  }

  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      preloadedImages.add(src)
      resolve()
    }
    img.onerror = () => resolve() // Resolve anyway to not block
    img.src = src
  })
}

// Export preload function for early loading
export const preloadPlatformAssets = (platform?: string) => {
  if (platform?.toLowerCase() === 'amd') {
    preloadImage('/amd.png')
  }
}

export const isImagePreloaded = (src: string) => preloadedImages.has(src)
export const preloadImageSrc = preloadImage
