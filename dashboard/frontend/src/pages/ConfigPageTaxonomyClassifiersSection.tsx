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
      eyebrow="Runtime"
      title="Taxonomy Classifiers"
      description="Inspect the router's built-in taxonomy package, manage custom classifier directories, and verify which tiers, categories, and taxonomy signals are wired to each classifier."
      configArea="Global"
      scope="Router-owned classifier packages and taxonomy signal bindings"
      panelEyebrow="Runtime"
      panelTitle="Classifier Catalog"
      panelDescription="These classifiers live in `global.model_catalog.classifiers[]`. Built-ins stay read-only, while custom classifiers can be created and edited from the dashboard."
      pills={[
        { label: 'Taxonomy Classifiers', active: true },
        { label: 'Global Config', active: false },
      ]}
    >
      <ConfigPageTaxonomyClassifiers
        isReadonly={isReadonly}
        openEditModal={openEditModal}
      />
    </ConfigPageManagerLayout>
  )
}
