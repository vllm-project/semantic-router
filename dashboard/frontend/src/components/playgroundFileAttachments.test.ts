import { describe, expect, it } from 'vitest'

import {
  buildPlaygroundUserContent,
  PLAYGROUND_MAX_INLINE_IMAGE_BYTES,
  readPlaygroundAttachmentFile,
  toPlaygroundAttachmentImages,
  validatePlaygroundAttachmentBudget,
} from './playgroundFileAttachments'

const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00])

describe('playgroundFileAttachments', () => {
  it('reads a verified raster image into a previewable data URL', async () => {
    const attachment = await readPlaygroundAttachmentFile(
      new File([pngBytes], 'architecture.png', { type: 'image/png' }),
    )

    expect(attachment).toMatchObject({
      fileName: 'architecture.png',
      kind: 'image',
      mediaType: 'image/png',
      sizeBytes: pngBytes.length,
    })
    expect(attachment.content).toMatch(/^data:image\/png;base64,/)
    expect(toPlaygroundAttachmentImages([attachment])).toEqual([
      { src: attachment.content, alt: 'architecture.png' },
    ])
  })

  it('rejects an image whose bytes do not match its declared type', async () => {
    await expect(
      readPlaygroundAttachmentFile(
        new File(['not an image'], 'architecture.png', { type: 'image/png' }),
      ),
    ).rejects.toThrow('does not contain a valid PNG image')
  })

  it('rejects active or unsupported image formats', async () => {
    await expect(
      readPlaygroundAttachmentFile(new File(['<svg/>'], 'diagram.svg', { type: 'image/svg+xml' })),
    ).rejects.toThrow('Use PNG, JPEG, GIF, or WebP')
  })

  it('builds text attachments and images into one OpenAI user content array', () => {
    expect(
      buildPlaygroundUserContent('Explain the attachment.', [
        {
          id: 'notes',
          fileName: 'notes.txt',
          sizeBytes: 5,
          content: 'hello',
          kind: 'text',
        },
        {
          id: 'image',
          fileName: 'architecture.png',
          sizeBytes: pngBytes.length,
          content: 'data:image/png;base64,AAAA',
          kind: 'image',
          mediaType: 'image/png',
        },
      ]),
    ).toEqual([
      {
        type: 'text',
        text: [
          'Explain the attachment.',
          '',
          `--- Attached file: notes.txt (5 B) ---`,
          'hello',
          '--- End of attached file ---',
        ].join('\n'),
      },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
    ])
  })

  it('rejects a multi-image set that would exceed the encoded request budget', () => {
    const content = `data:image/png;base64,${'A'.repeat(PLAYGROUND_MAX_INLINE_IMAGE_BYTES / 2)}`
    const image = {
      id: 'image-1',
      fileName: 'first.png',
      sizeBytes: 1,
      content,
      kind: 'image' as const,
      mediaType: 'image/png',
    }

    expect(validatePlaygroundAttachmentBudget([image])).toBeNull()
    expect(
      validatePlaygroundAttachmentBudget([
        image,
        { ...image, id: 'image-2', fileName: 'second.png' },
      ]),
    ).toContain('8.0 MB or less')
  })
})
