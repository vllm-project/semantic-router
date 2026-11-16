import type { FieldConfig } from '../components/EditModal'
import type { ViewSection } from '../components/ViewModal'

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
