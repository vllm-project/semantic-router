import React from 'react'
import Layout from '../components/Layout'
import type { ConfigSection } from '../components/ConfigNav'

export interface AppShellLayoutProps {
  configSection: ConfigSection
  setConfigSection: (section: ConfigSection) => void
  children: React.ReactNode
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

/** Wraps shared `Layout` with config-nav state owned by the route shell. */
const AppShellLayout: React.FC<AppShellLayoutProps> = ({
  configSection,
  setConfigSection,
  children,
  hideHeaderOnMobile,
  hideAccountControl,
}) => (
  <Layout
    configSection={configSection}
    onConfigSectionChange={(section) => setConfigSection(section as ConfigSection)}
    hideHeaderOnMobile={hideHeaderOnMobile}
    hideAccountControl={hideAccountControl}
  >
    {children}
  </Layout>
)

export default AppShellLayout
