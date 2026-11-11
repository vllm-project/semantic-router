import type { TurnUsage } from 'claude-code'

// Dollars per million tokens at list price, from the Claude API model table (2026-11): input, output, cache read.
// A cache write costs 1.25 times the input price. The first pattern that matches a model id or alias wins; a model
// matching none (a retired one, say) has no price, and its turns are left out of the estimate.
// Haiku 5.5 costs five times as much past a 100K-token prompt. A turn's usage is summed over its requests, so their
// prompt lengths are unknown: it is priced as the shorter prompts are, and a long one's saving is overstated.
type Price = { input: number; output: number; cacheRead: number }

const PRICES: readonly (readonly [RegExp, Price | undefined])[] = [
  [/fable-5-1|mythos/, { input: 10, output: 50, cacheRead: 0.25 }],
  [/fable/, { input: 10, output: 50, cacheRead: 1 }],
  [/opus-5-5|^opus$/, { input: 4, output: 20, cacheRead: 0.2 }],
  [/opus-4(-1)?(-\d{8})?$/, undefined],
  [/opus/, { input: 5, output: 25, cacheRead: 0.5 }],
  [/sonnet-5|^sonnet$/, { input: 2, output: 10, cacheRead: 0.2 }],
  [/sonnet/, { input: 3, output: 15, cacheRead: 0.3 }],
  [/haiku-5-5|^haiku$/, { input: 0.1, output: 0.5, cacheRead: 0.01 }],
  [/haiku/, { input: 1, output: 5, cacheRead: 0.1 }],
]

const priceOf = (model: string): Price | undefined => {
  const id = model.toLowerCase().replace(/\[.*\]$/, '')
  return PRICES.find(([pattern]) => pattern.test(id))?.[1]
}

// What `usage` costs on `model` at list price, in dollars; undefined for a model with no known price.
export function costOf(model: string, usage: TurnUsage): number | undefined {
  const price = priceOf(model)
  if (price === undefined) return undefined
  const dollars =
    usage.input_tokens * price.input +
    usage.output_tokens * price.output +
    usage.cache_read_input_tokens * price.cacheRead +
    usage.cache_creation_input_tokens * price.input * 1.25
  return dollars / 1_000_000
}
