import { createElement, isValidElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { Listener, VSRProfile } from '../types/config'
import ConfigPageListenerVSRPanel, { ListenerVSRControls } from './ConfigPageListenerVSRPanel'
import { withListenerVSR } from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

type Element = ReactElement<{ children?: unknown; onChange?: unknown; onClick?: unknown }>

const flatten = (node: unknown): Element[] => {
  if (Array.isArray(node)) return node.flatMap(flatten)
  if (!isValidElement(node)) return []
  const element = node as Element
  return [element, ...flatten(element.props.children)]
}

const text = (node: unknown): string => {
  if (typeof node === 'string') return node
  if (Array.isArray(node)) return node.map(text).join('')
  return isValidElement(node) ? text((node as Element).props.children) : ''
}

// The controls a person sees, with the profile they would write on a change.
function controls(vsr: VSRProfile | undefined) {
  const onChange = vi.fn<(next: VSRProfile | undefined) => void>()
  const tree = flatten(ListenerVSRControls({ vsr, disabled: false, onChange }))
  const field = (label: string) => {
    const owner = tree.find((element) => element.type === 'label' && text(element).includes(label))
    return flatten(owner?.props.children).find((element) => element.type !== 'span')
  }
  const button = (label: string) =>
    tree.find((element) => element.type === 'button' && text(element) === label)
  return { onChange, field, button }
}

const agentListener: Listener = {
  name: 'agents',
  address: '0.0.0.0',
  port: 8899,
  timeout: '300s',
  api_keys: ['client-key'],
  identity: { trust_headers: true },
  apis: {
    chat_completions: { models: ['auto'] },
    systemone: { schema: 'vllm-sr/5.0', max_deadline_ms: 5000, vsr: { guard: true } },
  },
}
const chatListener: Listener = {
  name: 'chat-only',
  address: '0.0.0.0',
  port: 8900,
  apis: { chat_completions: {} },
}
const config: ConfigData = { version: 'v0.5', listeners: [agentListener, chatListener] }

const writtenSystemOne = (next: ConfigData) => next.listeners?.[0].apis?.systemone

describe('vSR agent plugin profile on a System One listener', () => {
  it('writes a toggled checkbox to apis.systemone.vsr and keeps everything else', () => {
    const { onChange, field } = controls(agentListener.apis?.systemone?.vsr)
    const guard = field('Guard risky commands')
    expect(guard?.props).toMatchObject({ checked: true })
    ;(guard?.props.onChange as (event: unknown) => void)({ target: { checked: false } })
    const next = withListenerVSR(config, 0, onChange.mock.calls[0][0])

    expect(writtenSystemOne(next)).toEqual({
      schema: 'vllm-sr/5.0',
      max_deadline_ms: 5000,
      vsr: { guard: false },
    })
    expect({ ...next.listeners?.[0], apis: undefined }).toEqual({
      ...agentListener,
      apis: undefined,
    })
    expect(next.listeners?.[0].apis?.chat_completions).toEqual({ models: ['auto'] })
    expect(next.listeners?.[1]).toBe(chatListener)
    expect(next.version).toBe('v0.5')
  })

  it('shows an unset field at the plugin default and writes it explicitly on change', () => {
    const { onChange, field } = controls({ guard: true })
    expect(field('Route your own turns')?.props).toMatchObject({ checked: false })
    expect(field('Hide personal data (PII)')?.props).toMatchObject({ checked: true })
    expect(field('Router goal')?.props).toMatchObject({ value: 'balanced' })
    ;(field('Route your own turns')?.props.onChange as (event: unknown) => void)({
      target: { checked: true },
    })
    ;(field('Router goal')?.props.onChange as (event: unknown) => void)({
      target: { value: 'save' },
    })

    expect(onChange.mock.calls[0][0]).toEqual({ guard: true, router_turns: true })
    expect(onChange.mock.calls[1][0]).toEqual({ guard: true, router_goal: 'save' })
  })

  it('a change on a listener without vsr writes exactly that one field', () => {
    const bare = withListenerVSR(config, 0, undefined)
    const { onChange, field } = controls(undefined)
    expect(field('Route subagents')?.props).toMatchObject({ checked: true })
    ;(field('Route subagents')?.props.onChange as (event: unknown) => void)({
      target: { checked: false },
    })
    ;(field('Router goal')?.props.onChange as (event: unknown) => void)({
      target: { value: 'save' },
    })

    expect(writtenSystemOne(withListenerVSR(bare, 0, onChange.mock.calls[0][0]))).toEqual({
      schema: 'vllm-sr/5.0',
      max_deadline_ms: 5000,
      vsr: { router: false },
    })
    const next = withListenerVSR(bare, 0, onChange.mock.calls[1][0])
    expect(writtenSystemOne(next)?.vsr).toEqual({ router_goal: 'save' })
    expect(next.listeners?.[0].apis?.chat_completions).toEqual({ models: ['auto'] })
    expect(next.listeners?.[1]).toBe(chatListener)
  })

  it('"Clear" removes vsr and keeps the rest of the systemone entry', () => {
    const { onChange, button } = controls({ guard: true })
    expect(controls(undefined).button('Clear')).toBeUndefined()
    ;(button('Clear')?.props.onClick as () => void)()
    const next = withListenerVSR(config, 0, onChange.mock.calls[0][0])

    expect(writtenSystemOne(next)).toEqual({ schema: 'vllm-sr/5.0', max_deadline_ms: 5000 })
    expect(next.listeners?.[0].apis?.chat_completions).toEqual({ models: ['auto'] })
  })

  it('shows no vSR block for a listener without systemone', () => {
    const render = (listeners: Listener[]) =>
      renderToStaticMarkup(
        createElement(ConfigPageListenerVSRPanel, {
          config: { listeners },
          isReadonly: false,
          saveConfig: vi.fn(),
        }),
      )

    expect(render([chatListener])).toBe('')
    const markup = render([agentListener, chatListener])
    expect(markup).toContain('vSR agent plugins')
    expect(markup).toContain('agents')
    expect(markup).not.toContain('chat-only')
  })

  it('shows the plugin defaults, marked as defaults, when the listener has no profile', () => {
    const render = (next: ConfigData) =>
      renderToStaticMarkup(
        createElement(ConfigPageListenerVSRPanel, {
          config: next,
          isReadonly: false,
          saveConfig: vi.fn(),
        }),
      )
    const count = (markup: string, part: string) => markup.split(part).length - 1

    const bare = render(withListenerVSR(config, 0, undefined))
    expect(count(bare, 'type="checkbox"')).toBe(7)
    expect(count(bare, 'checked=""')).toBe(5)
    expect(count(bare, '>default<')).toBe(8)
    expect(bare).not.toContain('Clear')

    const profiled = render(config)
    expect(count(profiled, '>default<')).toBe(7)
    expect(profiled).toContain('Clear')
  })
})
