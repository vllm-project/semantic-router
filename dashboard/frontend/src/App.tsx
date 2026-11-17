import React, { useEffect, useState } from 'react'
import { ReadonlyProvider } from './contexts/ReadonlyContext'
import { CapabilitiesProvider } from './contexts/CapabilitiesContext'
import { SetupProvider } from './contexts/SetupContext'
import { AuthProvider } from './contexts/AuthContext'
import AppRouter from './app/AppRouter'
import { ThemeProvider, useTheme } from './shell/theme'
import { ShellProvider } from './shell/shell-context'
import { Button, Toaster, TooltipProvider } from './ui'

function ThemedToaster() {
  const { resolved } = useTheme()
  return <Toaster theme={resolved} />
}

function NestedDashboardNotice() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-background px-6 text-center text-foreground">
      <h1 className="text-xl font-semibold tracking-[-0.02em]">Nested Dashboard detected</h1>
      <p className="max-w-xl text-[13.5px] leading-relaxed text-muted-foreground">
        The Dashboard is running inside an iframe, which usually means a Grafana path or proxy
        points back at the Dashboard itself. Check the Grafana dashboard path and the backend proxy
        configuration.
      </p>
      <Button onClick={() => window.top?.location.reload()}>Open the Dashboard in a new tab</Button>
    </div>
  )
}

const App: React.FC = () => {
  const [isInIframe, setIsInIframe] = useState(false)

  useEffect(() => {
    if (window.self !== window.top) {
      setIsInIframe(true)
      console.warn('Dashboard detected it is running inside an iframe - this may indicate a loop')
    }
  }, [])

  return (
    <ThemeProvider>
      <ShellProvider>
        <TooltipProvider delayDuration={250} skipDelayDuration={150}>
          {isInIframe ? (
            <NestedDashboardNotice />
          ) : (
            <AuthProvider>
              <ReadonlyProvider>
                <CapabilitiesProvider>
                  <SetupProvider>
                    <AppRouter />
                  </SetupProvider>
                </CapabilitiesProvider>
              </ReadonlyProvider>
            </AuthProvider>
          )}
          <ThemedToaster />
        </TooltipProvider>
      </ShellProvider>
    </ThemeProvider>
  )
}

export default App
