import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { useConnectModelsDialogController } from './configPageConnectModelsDialogController'
import type { ConnectedModelInput } from './configPageConnectModelsDialogTypes'
import ConfigPageConnectModelsDialogView from './ConfigPageConnectModelsDialogView'

export type { ConnectedModelInput } from './configPageConnectModelsDialogTypes'

interface Props {
  isOpen: boolean
  existingModelNames: string[]
  reasoningFamilies: string[]
  catalog: BuiltInModelCatalog
  onClose: () => void
  onImport: (input: ConnectedModelInput) => Promise<void>
  onManualSetup: () => void
}

export default function ConfigPageConnectModelsDialog({
  isOpen,
  existingModelNames,
  reasoningFamilies,
  catalog,
  onClose,
  onImport,
  onManualSetup,
}: Props) {
  const controller = useConnectModelsDialogController(
    isOpen,
    existingModelNames,
    catalog,
    onClose,
    onImport,
  )
  if (!isOpen) return null
  return (
    <ConfigPageConnectModelsDialogView
      controller={controller}
      reasoningFamilies={reasoningFamilies}
      onClose={onClose}
      onManualSetup={onManualSetup}
    />
  )
}
