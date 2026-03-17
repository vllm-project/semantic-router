import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'
import type { ConfigData, Tool } from './configPageSupport'

export type OpenEditModal = <TForm extends object>(
  title: string,
  data: TForm,
  fields: FieldConfig<TForm>[],
  callback: (data: TForm) => Promise<void>,
  mode?: 'edit' | 'add'
) => void

export type OpenViewModal = (
  title: string,
  sections: ViewSection[],
  onEdit?: () => void
) => void

export type RouterConfigSectionData = Pick<
  ConfigData,
  'embedding_models' | 'semantic_cache' | 'tools' | 'prompt_guard' | 'classifier' | 'api' | 'observability'
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
