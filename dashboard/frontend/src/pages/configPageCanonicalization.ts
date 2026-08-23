import type { ConfigData } from './configPageSupport'

export const cloneConfigData = (value: ConfigData): ConfigData =>
  JSON.parse(JSON.stringify(value)) as ConfigData
