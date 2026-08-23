import type { ProviderCatalogIcon } from '../utils/providerCatalogApi'

const LOBE_ICON_VERSION = '1.90.0'
const LOBE_ICON_BASE = `https://unpkg.com/@lobehub/icons-static-svg@${LOBE_ICON_VERSION}/icons`

export const getProviderIconAsset = (icon?: ProviderCatalogIcon): string => {
  if (!icon) return ''
  if (icon.source !== 'lobe') return icon.value
  return `${LOBE_ICON_BASE}/${icon.value}${icon.color ? '-color' : ''}.svg`
}
