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
      configArea="Knowledge Base"
      scope="Router-owned KB packages and signal bindings"
      panelEyebrow="Manager"
      panelTitle="Knowledge Base Manager"
      panelDescription="This page owns knowledge base CRUD. Built-ins and custom KB packages can be updated through the same manager surface."
      pills={[
        { label: 'Knowledge Bases', active: true },
        { label: 'Groups', active: false },
        { label: 'Labels', active: false },
      ]}
    >
      <ConfigPageTaxonomyClassifiers
        isReadonly={isReadonly}
        openEditModal={openEditModal}
        activeView="knowledge-bases"
      />
    </ConfigPageManagerLayout>
  )
}
