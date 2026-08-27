import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageTaxonomyClassifiers from './ConfigPageTaxonomyClassifiers'
import type { OpenEditModal } from './configPageRouterSectionSupport'

interface ConfigPageTaxonomyClassifiersSectionProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
}

export default function ConfigPageTaxonomyClassifiersSection({
  isReadonly,
  openEditModal,
}: ConfigPageTaxonomyClassifiersSectionProps) {
  return (
    <ConfigPageManagerLayout
      eyebrow="Manager"
      title="Knowledge Bases"
      description="Manage router knowledge base packages in a dedicated surface: browse the KB catalog, inspect groups and labels, and update KB assets without mixing them into Global Config."
    >
      <ConfigPageTaxonomyClassifiers
        isReadonly={isReadonly}
        openEditModal={openEditModal}
        activeView="bases"
      />
    </ConfigPageManagerLayout>
  )
}
