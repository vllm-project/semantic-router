import React from 'react'
import Layout from '../components/Layout'

export interface AppShellLayoutProps {
  children: React.ReactNode
  hideHeaderOnMobile?: boolean
  hideAccountControl?: boolean
}

/** Wraps authenticated pages in the shared route-driven application shell. */
const AppShellLayout: React.FC<AppShellLayoutProps> = ({
  children,
  hideHeaderOnMobile,
  hideAccountControl,
}) => (
  <Layout hideHeaderOnMobile={hideHeaderOnMobile} hideAccountControl={hideAccountControl}>
    {children}
  </Layout>
)

export default AppShellLayout
