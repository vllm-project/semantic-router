import { describe, expect, it } from 'vitest'

import {
  PLAYGROUND_MAX_ATTACHMENT_BYTES,
  buildPromptWithAttachments,
  formatPlaygroundFileSize,
  playgroundAttachmentSizeError,
  readPlaygroundAttachmentFile,
  validatePlaygroundAttachmentSize,
} from './playgroundFileAttachments'

describe('playgroundFileAttachments', () => {
  it('formats file sizes for display', () => {
    expect(formatPlaygroundFileSize(512)).toBe('512 B')
    expect(formatPlaygroundFileSize(2048)).toBe('2.0 KB')
    expect(formatPlaygroundFileSize(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('rejects files larger than 10 MB before reading', () => {
    const oversized = new File(['x'], 'large.txt', { type: 'text/plain' })
    Object.defineProperty(oversized, 'size', { value: PLAYGROUND_MAX_ATTACHMENT_BYTES + 1 })

    expect(validatePlaygroundAttachmentSize(oversized)).toBe(
      playgroundAttachmentSizeError('large.txt', PLAYGROUND_MAX_ATTACHMENT_BYTES + 1)
    )
  })

  it('reads text file content', async () => {
    const file = new File(['hello world'], 'notes.txt', { type: 'text/plain' })
    const attachment = await readPlaygroundAttachmentFile(file)

    expect(attachment.fileName).toBe('notes.txt')
    expect(attachment.content).toBe('hello world')
    expect(attachment.sizeBytes).toBe(file.size)
  })

  it('builds a prompt that includes attached file content', () => {
    const prompt = buildPromptWithAttachments('Summarize this file', [{
      id: 'att-1',
      fileName: 'notes.txt',
      sizeBytes: 11,
      content: 'hello world',
    }])

    expect(prompt).toContain('Summarize this file')
    expect(prompt).toContain('--- Attached file: notes.txt (11 B) ---')
    expect(prompt).toContain('hello world')
    expect(prompt).toContain('--- End of attached file ---')
  })

  it('supports attachment-only prompts', () => {
    const prompt = buildPromptWithAttachments('', [{
      id: 'att-1',
      fileName: 'data.csv',
      sizeBytes: 4,
      content: 'a,b',
    }])

    expect(prompt).toContain('--- Attached file: data.csv (4 B) ---')
    expect(prompt).toContain('a,b')
  })
})
