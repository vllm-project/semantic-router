import type { InlineMessageImage } from './ChatComponentTypes'
import type { RecipeProbeMessage } from '../types/recipe'

export interface ProbeMessagePresentation {
  text: string
  images: InlineMessageImage[]
}

const INLINE_IMAGE_DATA_URL = /^data:image\/(?:gif|jpeg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/i
const MAX_INLINE_IMAGE_BYTES = 4 * 1024 * 1024
const TEXT_PART_TYPES = new Set(['input_text', 'output_text', 'text'])
const IMAGE_PART_TYPES = new Set(['image_url', 'input_image'])

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

const textPart = (value: unknown): string | null => {
  const item = asRecord(value)
  if (!item || typeof item.type !== 'string' || !TEXT_PART_TYPES.has(item.type)) return null
  return typeof item.text === 'string' ? item.text : null
}

const imageURL = (value: unknown): string | null => {
  const item = asRecord(value)
  if (!item || typeof item.type !== 'string' || !IMAGE_PART_TYPES.has(item.type)) return null
  const raw = item.image_url
  if (typeof raw === 'string') return raw
  const image = asRecord(raw)
  return typeof image?.url === 'string' ? image.url : null
}

export const isTrustedProbeImageSource = (source: string): boolean => {
  const match = source.match(INLINE_IMAGE_DATA_URL)
  if (!match) return false
  try {
    const decoded = atob(match[1])
    return (
      decoded.length > 0 && decoded.length <= MAX_INLINE_IMAGE_BYTES && btoa(decoded) === match[1]
    )
  } catch {
    return false
  }
}

export const assertTrustedProbeImages = (messages: RecipeProbeMessage[]): void => {
  messages.forEach((message, messageIndex) => {
    contentParts(message.content).forEach((part, contentIndex) => {
      const item = asRecord(part)
      if (!item || typeof item.type !== 'string' || !IMAGE_PART_TYPES.has(item.type)) return
      const source = imageURL(item)
      if (!source || !isTrustedProbeImageSource(source)) {
        throw new Error(
          `Probe message ${messageIndex + 1}, content part ${contentIndex + 1} contains an unverified image.`,
        )
      }
    })
  })
}

const contentParts = (content: unknown): unknown[] => (Array.isArray(content) ? content : [content])

export const presentProbeMessage = (message: RecipeProbeMessage): ProbeMessagePresentation => {
  if (typeof message.content === 'string') {
    return { text: message.content, images: [] }
  }
  if (message.content === null || message.content === undefined) {
    return { text: '', images: [] }
  }

  const text: string[] = []
  const images: InlineMessageImage[] = []
  let unsupportedParts = 0
  let unavailableImages = 0

  contentParts(message.content).forEach((part) => {
    const partText = textPart(part)
    if (partText !== null) {
      text.push(partText)
      return
    }

    const source = imageURL(part)
    if (source !== null) {
      if (isTrustedProbeImageSource(source)) {
        images.push({ src: source, alt: `Probe image ${images.length + 1}` })
      } else {
        unavailableImages += 1
      }
      return
    }

    unsupportedParts += 1
  })

  if (unavailableImages > 0) {
    text.push(`${unavailableImages} unverified image${unavailableImages === 1 ? '' : 's'} blocked.`)
  }
  if (unsupportedParts > 0) {
    text.push(
      `${unsupportedParts} structured content part${unsupportedParts === 1 ? '' : 's'} sent unchanged.`,
    )
  }
  return { text: text.join('\n\n'), images }
}

export const editableProbeMessageText = (
  message: RecipeProbeMessage | undefined,
): string | null => {
  if (!message) return null
  if (typeof message.content === 'string') return message.content
  if (!Array.isArray(message.content)) return null
  const textParts = message.content.map(textPart).filter((value): value is string => value !== null)
  return textParts.length === 1 ? textParts[0] : null
}

export const replaceEditableProbeMessageText = (
  message: RecipeProbeMessage,
  nextText: string,
): RecipeProbeMessage => {
  if (typeof message.content === 'string') return { ...message, content: nextText }
  if (!Array.isArray(message.content)) {
    throw new Error('This structured probe does not contain one editable text part.')
  }

  let replaced = false
  const content = message.content.map((part) => {
    if (replaced || textPart(part) === null) return part
    replaced = true
    return { ...(asRecord(part) ?? {}), text: nextText }
  })
  if (!replaced || message.content.filter((part) => textPart(part) !== null).length !== 1) {
    throw new Error('This structured probe does not contain one editable text part.')
  }
  return { ...message, content }
}
