import { describe, expect, it } from 'vitest'

import {
  createProbePlaygroundTask,
  materializeEditedProbeRequest,
  preparePlaygroundInvocation,
  resolveProbeRequestModel,
  toPlaygroundHistory,
} from './playgroundInvocationSupport'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'

const invocation = (intent: 'run' | 'edit'): PlaygroundInvocation => ({
  version: 1,
  intent,
  source: 'recipe-probe',
  probeId: 'balanced_recovery/semantic_reask',
  recipeDigest: 'sha256:recipe',
  recipe: 'balanced',
  editable: true,
  model: 'team/custom-balanced',
  messages: [
    { role: 'user', content: 'Explain vector clocks.' },
    { role: 'assistant', content: 'A vector clock is a logical timestamp.' },
    { role: 'user', content: 'Explain vector clocks.' },
  ],
  tools: [{ type: 'function', function: { name: 'lookup', parameters: {} } }],
  toolChoice: { type: 'function', function: { name: 'lookup' } },
  request: { temperature: 0 },
})

describe('playgroundInvocationSupport', () => {
  it('resolves a model-free probe through the active Recipe entrypoint', () => {
    const prepared = preparePlaygroundInvocation({
      ...invocation('run'),
      model: 'vllm-sr/auto',
    })

    expect(
      resolveProbeRequestModel(prepared, [
        { id: 'vllm-sr/mom-v1-lite', description: '', recipe: 'cost' },
        { id: 'vllm-sr/mom-v1-blend', description: '', recipe: 'balanced' },
      ]),
    ).toBe('vllm-sr/mom-v1-blend')
  })

  it('prepares the final user turn as the editable prompt', () => {
    const prepared = preparePlaygroundInvocation(invocation('edit'))

    expect(prepared.prompt).toBe('Explain vector clocks.')
    expect(prepared.history).toHaveLength(2)
    expect(prepared.exactRequest).toMatchObject({
      model: 'team/custom-balanced',
      temperature: 0,
      tools: expect.any(Array),
      tool_choice: { type: 'function', function: { name: 'lookup' } },
      messages: expect.any(Array),
    })
  })

  it('replaces only the editable turn and permits a selected model change', () => {
    const prepared = preparePlaygroundInvocation(invocation('edit'))

    const request = materializeEditedProbeRequest(
      prepared,
      'Explain them with two concurrent writers.',
      'team/custom-fast',
    )

    expect(request.model).toBe('team/custom-fast')
    expect(request.messages).toEqual([
      { role: 'user', content: 'Explain vector clocks.' },
      { role: 'assistant', content: 'A vector clock is a logical timestamp.' },
      { role: 'user', content: 'Explain them with two concurrent writers.' },
    ])
  })

  it('creates display history without changing the exact request', () => {
    const prepared = preparePlaygroundInvocation(invocation('run'))
    let nextId = 0

    const history = toPlaygroundHistory(prepared.history, () => `message-${nextId++}`)

    expect(history.map((message) => message.role)).toEqual(['user', 'assistant'])
    expect(history.map((message) => message.content)).toEqual([
      'Explain vector clocks.',
      'A vector clock is a logical timestamp.',
    ])
  })

  it('builds an isolated task that sends the probe tools without executing them', () => {
    const task = createProbePlaygroundTask(
      preparePlaygroundInvocation(invocation('run')),
      'conversation-1',
      'team/custom-balanced',
    )

    expect(task).toMatchObject({
      conversationId: 'conversation-1',
      prompt: 'Explain vector clocks.',
      requestOptions: {
        enableClawMode: false,
        enableWebSearch: false,
        executeToolCalls: false,
        model: 'team/custom-balanced',
      },
      exactRequest: {
        model: 'team/custom-balanced',
        temperature: 0,
        tools: expect.any(Array),
        tool_choice: { type: 'function', function: { name: 'lookup' } },
        messages: expect.any(Array),
      },
      appendPromptMessage: true,
    })
  })

  it('renders multimodal content as text and an image without exposing the data URI', () => {
    const image = 'data:image/png;base64,AA=='
    const structured: PlaygroundInvocation = {
      ...invocation('run'),
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Summarize this diagram.' },
            { type: 'image_url', image_url: { url: image } },
          ],
        },
      ],
    }

    const prepared = preparePlaygroundInvocation(structured)
    const task = createProbePlaygroundTask(prepared, 'conversation-image', structured.model!)

    expect(prepared.prompt).toBe('Summarize this diagram.')
    expect(prepared.prompt).not.toContain('data:image')
    expect(prepared.promptImages).toEqual([{ src: image, alt: 'Probe image 1' }])
    expect(task.displayMessage).toEqual({
      content: 'Summarize this diagram.',
      images: [{ src: image, alt: 'Probe image 1' }],
    })
    expect(task.exactRequest?.messages).toEqual(structured.messages)
  })

  it('edits the single text part of a multimodal request without dropping its image', () => {
    const imagePart = {
      type: 'image_url',
      image_url: { url: 'data:image/png;base64,AA==' },
    }
    const structured: PlaygroundInvocation = {
      ...invocation('edit'),
      messages: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Original question.' }, imagePart],
        },
      ],
    }

    const request = materializeEditedProbeRequest(
      preparePlaygroundInvocation(structured),
      'Edited question.',
      structured.model!,
    )

    expect(request.messages).toEqual([
      {
        role: 'user',
        content: [{ type: 'text', text: 'Edited question.' }, imagePart],
      },
    ])
  })

  it('adds an uploaded image to an edited probe without dropping the fixture image', () => {
    const fixtureImage = {
      type: 'image_url',
      image_url: { url: 'data:image/png;base64,AA==' },
    }
    const uploadedImage = {
      id: 'uploaded',
      fileName: 'new.png',
      sizeBytes: 1,
      content: 'data:image/png;base64,AQ==',
      kind: 'image' as const,
      mediaType: 'image/png',
    }
    const structured: PlaygroundInvocation = {
      ...invocation('edit'),
      messages: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Original question.' }, fixtureImage],
        },
      ],
    }

    const request = materializeEditedProbeRequest(
      preparePlaygroundInvocation(structured),
      'Edited question.',
      structured.model!,
      [uploadedImage],
    )

    expect(request.messages).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Edited question.' },
          fixtureImage,
          { type: 'image_url', image_url: { url: uploadedImage.content } },
        ],
      },
    ])
  })

  it('rejects an external probe image instead of sending it to the model backend', () => {
    const structured: PlaygroundInvocation = {
      ...invocation('edit'),
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Inspect this image.' },
            { type: 'image_url', image_url: { url: 'https://private.example/image.png' } },
          ],
        },
      ],
    }

    expect(() => preparePlaygroundInvocation(structured)).toThrow('contains an unverified image')
  })

  it('renders a multimodal history turn without changing the terminal request', () => {
    const image = 'data:image/webp;base64,AA=='
    const assistantImage = 'data:image/png;base64,AA=='
    const structured: PlaygroundInvocation = {
      ...invocation('run'),
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Earlier image.' },
            { type: 'image_url', image_url: { url: image } },
          ],
        },
        {
          role: 'assistant',
          content: [
            { type: 'output_text', text: 'Earlier answer.' },
            { type: 'image_url', image_url: assistantImage },
          ],
        },
        { role: 'user', content: 'Final question.' },
      ],
    }

    const prepared = preparePlaygroundInvocation(structured)
    const history = toPlaygroundHistory(prepared.history, () => crypto.randomUUID())

    expect(history[0]).toMatchObject({
      role: 'user',
      content: 'Earlier image.',
      images: [{ src: image, alt: 'Probe image 1' }],
    })
    expect(history[1]).toMatchObject({
      role: 'assistant',
      content: 'Earlier answer.',
      images: [{ src: assistantImage, alt: 'Probe image 1' }],
    })
    expect(prepared.exactRequest.messages).toEqual(structured.messages)
  })

  it('preserves a terminal tool result as history without inventing a user turn', () => {
    const structured: PlaygroundInvocation = {
      ...invocation('run'),
      editable: false,
      messages: [
        { role: 'user', content: 'Look up the policy.' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call-1',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
        { role: 'tool', tool_call_id: 'call-1', content: 'Policy result.' },
      ],
    }

    const prepared = preparePlaygroundInvocation(structured)
    const history = toPlaygroundHistory(prepared.history, () => crypto.randomUUID())
    const task = createProbePlaygroundTask(prepared, 'conversation-2', structured.model!)

    expect(prepared.editable).toBe(false)
    expect(prepared.prompt).toBe('')
    expect(history).toHaveLength(2)
    expect(history[1]).toMatchObject({
      role: 'assistant',
      toolCalls: [{ id: 'call-1', status: 'completed' }],
      toolResults: [{ callId: 'call-1', content: 'Policy result.' }],
    })
    expect(task.appendPromptMessage).toBe(false)
  })
})
