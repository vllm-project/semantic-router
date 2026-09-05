import ConfigPageModelsSectionView from './ConfigPageModelsSectionView'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'
import ModelDeleteDialog from './ModelDeleteDialog'
import { useConfigPageModelsSectionController } from './useConfigPageModelsSectionController'

export default function ConfigPageModelsSection(props: ConfigPageModelsSectionProps) {
  const controller = useConfigPageModelsSectionController(props)
  const modelsPendingDelete = controller.deletion.modelNames
  return (
    <>
      <ConfigPageModelsSectionView props={props} controller={controller} />
      <ModelDeleteDialog
        modelNames={modelsPendingDelete}
        pending={controller.deletion.pending}
        onCancel={() => controller.deletion.setModelNames([])}
        onConfirm={() => void controller.deletion.remove(modelsPendingDelete)}
      />
    </>
  )
}
