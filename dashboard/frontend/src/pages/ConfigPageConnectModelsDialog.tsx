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
  /** A catalog model proposed from the Model Hub; its providers are offered first. */
  catalogModel?: string
  onClose: () => void
  onImport: (input: ConnectedModelInput) => Promise<void>
  onManualSetup: () => void
}

export default function ConfigPageConnectModelsDialog({
  isOpen,
  existingModelNames,
  reasoningFamilies,
  catalog,
  catalogModel,
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
    catalogModel,
  )
  return (
    <ConfigPageConnectModelsDialogView
      isOpen={isOpen}
      controller={controller}
      reasoningFamilies={reasoningFamilies}
      onClose={onClose}
      onManualSetup={onManualSetup}
    />
  )
}
