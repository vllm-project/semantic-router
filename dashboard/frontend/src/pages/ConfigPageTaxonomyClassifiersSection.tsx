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
      title="Taxonomy Classifiers"
      description="Manage taxonomy classifier packages in a dedicated surface: browse the classifier catalog, inspect tiers and categories, and update custom classifiers without mixing them into Global Config."
      configArea="Classifiers"
      scope="Router-owned taxonomy classifier packages and signal bindings"
      panelEyebrow="Manager"
      panelTitle="Classifier Manager"
      panelDescription="This page owns taxonomy classifier CRUD. Built-ins stay visible and read-only; custom classifier directories can be created and updated independently."
      pills={[
        { label: 'Classifiers', active: true },
        { label: 'Tiers', active: false },
        { label: 'Categories', active: false },
      ]}
    >
      <ConfigPageTaxonomyClassifiers
        isReadonly={isReadonly}
        openEditModal={openEditModal}
      />
    </ConfigPageManagerLayout>
  )
}
