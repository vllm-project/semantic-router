const PLAYGROUND_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
const PLAYGROUND_MAX_IMAGE_BYTES = 4 * 1024 * 1024
export const PLAYGROUND_MAX_INLINE_IMAGE_BYTES = 8 * 1024 * 1024
const IMAGE_MEDIA_TYPES = new Set(['image/gif', 'image/jpeg', 'image/png', 'image/webp'])
const IMAGE_EXTENSION_TYPES: Record<string, string> = {
  gif: 'image/gif',
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
}

export interface PlaygroundAttachment {
  id: string
  fileName: string
  sizeBytes: number
  content: string
  kind?: 'image' | 'text'
  mediaType?: string
}

export type PlaygroundAttachmentSummary = Pick<PlaygroundAttachment, 'fileName' | 'sizeBytes'>

export const generatePlaygroundAttachmentId = (): string =>
  `att-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`

export const formatPlaygroundFileSize = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '0 B'
  }
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0)} MB`
}

export type PlaygroundUserContent =
  | string
  | Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }>

const playgroundAttachmentSizeError = (
  fileName: string,
  sizeBytes: number,
  limitBytes: number,
): string => {
  const limitLabel = formatPlaygroundFileSize(limitBytes)
  const actualLabel = formatPlaygroundFileSize(sizeBytes)
  return `"${fileName}" is ${actualLabel}. Each attachment must be ${limitLabel} or smaller.`
}

const fileExtension = (fileName: string): string => fileName.split('.').pop()?.toLowerCase() ?? ''

const declaredImageMediaType = (file: File): string | null => {
  const mediaType = file.type.toLowerCase()
  if (IMAGE_MEDIA_TYPES.has(mediaType)) return mediaType
  return IMAGE_EXTENSION_TYPES[fileExtension(file.name)] ?? null
}

const hasImageIntent = (file: File): boolean =>
  file.type.toLowerCase().startsWith('image/') || fileExtension(file.name) in IMAGE_EXTENSION_TYPES

const validatePlaygroundAttachmentSize = (file: File, limitBytes: number): string | null => {
  if (file.size > limitBytes) {
    return playgroundAttachmentSizeError(file.name, file.size, limitBytes)
  }
  return null
}

const sniffImageMediaType = (bytes: Uint8Array): string | null => {
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return 'image/png'
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg'
  }
  const prefix = String.fromCharCode(...bytes.slice(0, 12))
  if (prefix.startsWith('GIF87a') || prefix.startsWith('GIF89a')) return 'image/gif'
  if (prefix.startsWith('RIFF') && prefix.slice(8, 12) === 'WEBP') return 'image/webp'
  return null
}

const bytesToBase64 = (bytes: Uint8Array): string => {
  let binary = ''
  const chunkSize = 0x8000
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize))
  }
  return btoa(binary)
}

export const isPlaygroundImageAttachment = (attachment: PlaygroundAttachment): boolean =>
  attachment.kind === 'image'

export const validatePlaygroundAttachmentBudget = (
  attachments: PlaygroundAttachment[],
): string | null => {
  const encodedImageBytes = attachments
    .filter(isPlaygroundImageAttachment)
    .reduce((total, attachment) => total + attachment.content.length, 0)
  if (encodedImageBytes <= PLAYGROUND_MAX_INLINE_IMAGE_BYTES) return null
  return `Images in one request must use ${formatPlaygroundFileSize(PLAYGROUND_MAX_INLINE_IMAGE_BYTES)} or less after encoding.`
}

export const readPlaygroundAttachmentFile = async (file: File): Promise<PlaygroundAttachment> => {
  const requestedImageType = declaredImageMediaType(file)
  if (hasImageIntent(file) && !requestedImageType) {
    throw new Error(`"${file.name}" is not a supported image. Use PNG, JPEG, GIF, or WebP.`)
  }

  const limitBytes = requestedImageType
    ? PLAYGROUND_MAX_IMAGE_BYTES
    : PLAYGROUND_MAX_ATTACHMENT_BYTES
  const sizeError = validatePlaygroundAttachmentSize(file, limitBytes)
  if (sizeError) {
    throw new Error(sizeError)
  }

  try {
    if (requestedImageType) {
      const bytes = new Uint8Array(await file.arrayBuffer())
      const detectedImageType = sniffImageMediaType(bytes)
      if (!detectedImageType || detectedImageType !== requestedImageType) {
        throw new Error(
          `"${file.name}" does not contain a valid ${requestedImageType.replace('image/', '').toUpperCase()} image.`,
        )
      }
      return {
        id: generatePlaygroundAttachmentId(),
        fileName: file.name,
        sizeBytes: file.size,
        content: `data:${detectedImageType};base64,${bytesToBase64(bytes)}`,
        kind: 'image',
        mediaType: detectedImageType,
      }
    }

    const content = await file.text()
    return {
      id: generatePlaygroundAttachmentId(),
      fileName: file.name,
      sizeBytes: file.size,
      content,
      kind: 'text',
    }
  } catch (error) {
    if (error instanceof Error && error.message.includes(file.name)) throw error
    throw new Error(`Could not read "${file.name}". Try a text-based file.`)
  }
}

export const buildPromptWithAttachments = (
  prompt: string,
  attachments: PlaygroundAttachment[] = [],
): string => {
  const trimmedPrompt = prompt.trim()
  const textAttachments = attachments.filter(
    (attachment) => !isPlaygroundImageAttachment(attachment),
  )
  if (textAttachments.length === 0) {
    return trimmedPrompt
  }

  const attachmentBlocks = textAttachments.map((attachment) =>
    [
      `--- Attached file: ${attachment.fileName} (${formatPlaygroundFileSize(attachment.sizeBytes)}) ---`,
      attachment.content,
      '--- End of attached file ---',
    ].join('\n'),
  )

  const attachmentSection = attachmentBlocks.join('\n\n')
  if (!trimmedPrompt) {
    return attachmentSection
  }

  return `${trimmedPrompt}\n\n${attachmentSection}`
}

export const buildPlaygroundUserContent = (
  prompt: string,
  attachments: PlaygroundAttachment[] = [],
): PlaygroundUserContent => {
  const text = buildPromptWithAttachments(prompt, attachments)
  const images = attachments.filter(isPlaygroundImageAttachment)
  if (images.length === 0) return text

  return [
    ...(text ? [{ type: 'text' as const, text }] : []),
    ...images.map((attachment) => ({
      type: 'image_url' as const,
      image_url: { url: attachment.content },
    })),
  ]
}

export const toPlaygroundAttachmentImages = (attachments: PlaygroundAttachment[]) =>
  attachments.filter(isPlaygroundImageAttachment).map((attachment) => ({
    src: attachment.content,
    alt: attachment.fileName,
  }))

export const toPlaygroundAttachmentSummaries = (
  attachments: PlaygroundAttachment[],
): PlaygroundAttachmentSummary[] =>
  attachments.map(({ fileName, sizeBytes }) => ({ fileName, sizeBytes }))
