import ConfigPageClassifierSection from './ConfigPageClassifierSection'
import ConfigPageLegacyCategoriesSection from './ConfigPageLegacyCategoriesSection'
import ConfigPageSafetyCacheSection from './ConfigPageSafetyCacheSection'
import ConfigPageToolsObservabilitySection from './ConfigPageToolsObservabilitySection'
import type { RouterToolsSectionProps } from './configPageRouterSectionSupport'
import type { LegacyCategoriesConfig } from './configPageCanonicalState'

interface ConfigPageRouterConfigSectionProps extends RouterToolsSectionProps {
  legacyCategoriesConfig?: LegacyCategoriesConfig | null
}

export default function ConfigPageRouterConfigSection({
  config,
  routerConfig,
  toolsData,
  toolsLoading,
  toolsError,
  isReadonly,
  openEditModal,
  saveConfig,
  legacyCategoriesConfig = null,
}: ConfigPageRouterConfigSectionProps) {
  const baseProps = {
    config,
    routerConfig,
    isReadonly,
    openEditModal,
    saveConfig,
  }

  return (
    <div>
      <ConfigPageSafetyCacheSection {...baseProps} />
      <ConfigPageClassifierSection {...baseProps} />
      <ConfigPageToolsObservabilitySection
        {...baseProps}
        toolsData={toolsData}
        toolsLoading={toolsLoading}
        toolsError={toolsError}
      />
      {legacyCategoriesConfig ? (
        <ConfigPageLegacyCategoriesSection
          legacyConfig={legacyCategoriesConfig}
          isReadonly={isReadonly}
          openEditModal={openEditModal}
          saveConfig={saveConfig}
        />
      ) : null}
    </div>
  )
}
