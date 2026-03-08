import type { FieldConfig } from '../components/EditModal'
import type { ConfigData, Tool } from './configPageSupport'

export type OpenEditModal = (
  title: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any,
  fields: FieldConfig[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  callback: (data: any) => Promise<void>,
  mode?: 'edit' | 'add'
) => void

export type RouterConfigSectionData = Pick<
  ConfigData,
  'bert_model' | 'semantic_cache' | 'tools' | 'prompt_guard' | 'classifier' | 'api' | 'observability'
>

export interface RouterSectionBaseProps {
  config: ConfigData | null
  routerConfig: RouterConfigSectionData
  isReadonly: boolean
  openEditModal: OpenEditModal
  saveConfig: (config: ConfigData) => Promise<void>
}

export interface RouterToolsSectionProps extends RouterSectionBaseProps {
  toolsData: Tool[]
  toolsLoading: boolean
  toolsError: string | null
}

export interface LegacyCategoriesSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  openEditModal: OpenEditModal
  saveConfig: (config: ConfigData) => Promise<void>
}

export const cloneConfig = (config: ConfigData | null): ConfigData => ({ ...(config || {}) })
