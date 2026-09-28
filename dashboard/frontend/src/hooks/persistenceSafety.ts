const INLINE_IMAGE_PREFIX = /^data:image\/(?:gif|jpeg|png|webp);base64,/i

export const containsInlineImageData = (value: unknown, seen = new WeakSet<object>()): boolean => {
  if (typeof value === 'string') return INLINE_IMAGE_PREFIX.test(value)
  if (!value || typeof value !== 'object') return false
  if (seen.has(value)) return false
  seen.add(value)
  if (Array.isArray(value)) {
    return value.some((item) => containsInlineImageData(item, seen))
  }
  return Object.values(value).some((item) => containsInlineImageData(item, seen))
}

export const jsonByteLength = (value: unknown): number =>
  new TextEncoder().encode(JSON.stringify(value)).byteLength
