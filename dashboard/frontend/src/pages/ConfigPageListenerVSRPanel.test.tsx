import { createElement, isValidElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { Listener, VSRProfile } from '../types/config'
import ConfigPageListenerVSRPanel, { ListenerVSRControls } from './ConfigPageListenerVSRPanel'
import { withListenerVSR } from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

type Element = ReactElement<{
  children?: unknown
  label?: string
  choices?: Array<{ text: string; pressed: boolean; pick: () => void }>
}>

const flatten = (node: unknown): Element[] => {
  if (Array.isArray(node)) return node.flatMap(flatten)
  if (!isValidElement(node)) return []
  const element = node as Element
  return [element, ...flatten(element.props.children)]
}

type Button = ReactElement<{ children?: unknown; disabled?: boolean; onClick?: () => void }>

// The buttons one row draws, rendered from its Choices element as React would.
const buttonsOf = (choices: Element) =>
  flatten((choices.type as (props: unknown) => unknown)(choices.props)).filter(
    (element) => element.type === 'button',
  ) as Button[]

const text = (node: unknown): string =>
  typeof node === 'string'
    ? node
    : Array.isArray(node)
      ? node.map(text).join('')
      : isValidElement(node)
        ? text((node as Element).props.children)
        : ''

// The choices a person sees on one row, and the profile a click on one writes.
function controls(vsr: VSRProfile | undefined, disabled = false) {
  const onChange = vi.fn<(next: VSRProfile | undefined) => void>()
  const tree = flatten(ListenerVSRControls({ vsr, disabled, onChange }))
  const row = (label: string) => {
    const element = tree.find((one) => one.props.label === label && one.props.choices)
    const buttons = element ? buttonsOf(element) : []
    const named = (name: string) => buttons.find((button) => text(button.props.children).startsWith(name))
    return {
      buttons,
      pressed: element?.props.choices?.find((choice) => choice.pressed)?.text,
      pick: (name: string) => named(name)?.props.onClick?.(),
    }
  }
  return { onChange, row }
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
  it('On or Off writes that field to apis.systemone.vsr and keeps everything else', () => {
    const { onChange, row } = controls(agentListener.apis?.systemone?.vsr)
    expect(row('Guard risky commands').pressed).toBe('On')
    row('Guard risky commands').pick('Off')
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

  it('an unset field has Default pressed and writes only itself on a pick', () => {
    const { onChange, row } = controls({ guard: true })
    expect(row('Route your own turns').pressed).toBe('Default')
    expect(row('Router goal').pressed).toBe('Default')
    row('Route your own turns').pick('On')
    row('Router goal').pick('Save')

    expect(onChange.mock.calls[0][0]).toEqual({ guard: true, router_turns: true })
    expect(onChange.mock.calls[1][0]).toEqual({ guard: true, router_goal: 'save' })
  })

  it('Default takes a field back out, and the last one takes vsr out of systemone', () => {
    const { onChange, row } = controls({ guard: false, router_goal: 'save' })
    row('Router goal').pick('Default')
    expect(onChange.mock.calls[0][0]).toEqual({ guard: false })

    const last = controls({ guard: false })
    last.row('Guard risky commands').pick('Default')
    expect(last.onChange.mock.calls[0][0]).toBeUndefined()
    expect(writtenSystemOne(withListenerVSR(config, 0, undefined))).toEqual({
      schema: 'vllm-sr/5.0',
      max_deadline_ms: 5000,
    })
  })

  it('every button of a row is disabled while the panel saves', () => {
    const { row } = controls({ guard: true }, true)
    expect(row('Router goal').buttons.map((button) => button.props.disabled)).toEqual([
      true,
      true,
      true,
      true,
    ])
  })

  it('a pick on a listener without vsr writes exactly that one field', () => {
    const bare = withListenerVSR(config, 0, undefined)
    const { onChange, row } = controls(undefined)
    row('Route subagents').pick('Off')
    const next = withListenerVSR(bare, 0, onChange.mock.calls[0][0])

    expect(writtenSystemOne(next)?.vsr).toEqual({ router: false })
    expect(next.listeners?.[1]).toBe(chatListener)
  })

  it('a pick and a Default keep the listener\'s rules', () => {
    const rules = { recipe: 'default', guard: [{ type: 'keyword' as const, name: 'prod_database', enforce: 'required' as const }] }
    const { onChange, row } = controls({ guard: true, rules })
    row('Guard risky commands').pick('Off')
    expect(onChange.mock.calls[0][0]).toEqual({ guard: false, rules })
    row('Guard risky commands').pick('Default')
    expect(onChange.mock.calls[1][0]).toEqual({ rules })
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

  it('says how many fields the listener sets, and offers a reset only when it sets one', () => {
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
    expect(count(bare, 'aria-pressed="true">Default')).toBe(8)
    expect(bare).toContain('All plugin defaults')
    expect(bare).not.toContain('Reset to plugin defaults')

    const profiled = render(config)
    expect(count(profiled, 'aria-pressed="true">Default')).toBe(7)
    expect(profiled).toContain('1 set here')
    expect(profiled).toContain('Reset to plugin defaults')
  })
})
