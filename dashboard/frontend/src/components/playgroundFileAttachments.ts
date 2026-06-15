const PLAYGROUND_MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024

export interface PlaygroundAttachment {
  id: string
  fileName: string
  sizeBytes: number
  content: string
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

const playgroundAttachmentSizeError = (fileName: string, sizeBytes: number): string => {
  const limitLabel = formatPlaygroundFileSize(PLAYGROUND_MAX_ATTACHMENT_BYTES)
  const actualLabel = formatPlaygroundFileSize(sizeBytes)
  return `"${fileName}" is ${actualLabel}. Each attachment must be ${limitLabel} or smaller.`
}

const validatePlaygroundAttachmentSize = (file: File): string | null => {
  if (file.size > PLAYGROUND_MAX_ATTACHMENT_BYTES) {
    return playgroundAttachmentSizeError(file.name, file.size)
  }
  return null
}

export const readPlaygroundAttachmentFile = async (file: File): Promise<PlaygroundAttachment> => {
  const sizeError = validatePlaygroundAttachmentSize(file)
  if (sizeError) {
    throw new Error(sizeError)
  }

  try {
    const content = await file.text()
    return {
      id: generatePlaygroundAttachmentId(),
      fileName: file.name,
      sizeBytes: file.size,
      content,
    }
  } catch {
    throw new Error(`Could not read "${file.name}". Try a text-based file.`)
  }
}

export const buildPromptWithAttachments = (
  prompt: string,
  attachments: PlaygroundAttachment[] = []
): string => {
  const trimmedPrompt = prompt.trim()
  if (attachments.length === 0) {
    return trimmedPrompt
  }

  const attachmentBlocks = attachments.map(attachment => [
    `--- Attached file: ${attachment.fileName} (${formatPlaygroundFileSize(attachment.sizeBytes)}) ---`,
    attachment.content,
    '--- End of attached file ---',
  ].join('\n'))

  const attachmentSection = attachmentBlocks.join('\n\n')
  if (!trimmedPrompt) {
    return attachmentSection
  }

  return `${trimmedPrompt}\n\n${attachmentSection}`
}

export const toPlaygroundAttachmentSummaries = (
  attachments: PlaygroundAttachment[]
): PlaygroundAttachmentSummary[] =>
  attachments.map(({ fileName, sizeBytes }) => ({ fileName, sizeBytes }))
