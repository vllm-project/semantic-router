import ConfigPageClassifierSection from './ConfigPageClassifierSection'
import ConfigPageLegacyCategoriesSection from './ConfigPageLegacyCategoriesSection'
import ConfigPageSafetyCacheSection from './ConfigPageSafetyCacheSection'
import ConfigPageToolsObservabilitySection from './ConfigPageToolsObservabilitySection'
import type { RouterToolsSectionProps } from './configPageRouterSectionSupport'

interface ConfigPageRouterConfigSectionProps extends RouterToolsSectionProps {
  showLegacyCategories?: boolean
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
  showLegacyCategories = false,
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
      {showLegacyCategories ? (
        <ConfigPageLegacyCategoriesSection
          config={config}
          isReadonly={isReadonly}
          openEditModal={openEditModal}
          saveConfig={saveConfig}
        />
      ) : null}
    </div>
  )
}
