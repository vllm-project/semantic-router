import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { PROVIDER_LB_POLICIES } from '../generated/routerConfigContract'
import ConfigPageConnectModelAdvanced from './ConfigPageConnectModelAdvanced'
import { emptyConnectModelAdvancedValues } from './configPageConnectModelSupport'

describe('Quick Connect delivery options', () => {
  it('renders every generated policy rather than a parallel option list', () => {
    const html = renderToStaticMarkup(
      <ConfigPageConnectModelAdvanced
        value={emptyConnectModelAdvancedValues()}
        reasoningFamilies={[]}
        onChange={() => {}}
      />,
    )
    const selects = [...html.matchAll(/<select[^>]*>([\s\S]*?)<\/select>/g)]
    const select = selects[selects.length - 1]?.[1] ?? ''
    const values = [...select.matchAll(/<option value="([^"]*)"/g)].map((match) => match[1])
    expect(values).toEqual(['', ...PROVIDER_LB_POLICIES])
  })
})
