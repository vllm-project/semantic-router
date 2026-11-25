import React from 'react'
import { ArrowUpRight, RefreshCw, Unplug } from 'lucide-react'

import { Button, Card, CodeBlock, EmptyState } from '../ui'

export interface ServiceConfig {
  name: string
  envVar: string
  description: string
  docsUrl?: string
  exampleValue?: string
}

interface ServiceNotConfiguredProps {
  service: ServiceConfig
  onRetry?: () => void
}

const ServiceNotConfigured: React.FC<ServiceNotConfiguredProps> = ({ service, onRetry }) => {
  const assignment = `${service.envVar}=${service.exampleValue ?? `<${service.name.toLowerCase()}-url>`}`

  return (
    <Card className="border-dashed shadow-none">
      <EmptyState
        icon={Unplug}
        tone="blue"
        title={`Connect ${service.name}`}
        description={
          <>
            {service.description} Set{' '}
            <code className="font-mono text-[12px]">{service.envVar}</code> for the dashboard to
            enable it.
          </>
        }
        actions={
          <>
            {onRetry ? (
              <Button type="button" onClick={onRetry}>
                <RefreshCw />
                Retry connection
              </Button>
            ) : null}
            {service.docsUrl ? (
              <Button variant="outline" asChild>
                <a href={service.docsUrl} target="_blank" rel="noopener noreferrer">
                  View documentation
                  <ArrowUpRight />
                </a>
              </Button>
            ) : null}
          </>
        }
      />
      <div className="mx-auto w-full max-w-2xl px-6 pb-12">
        <CodeBlock
          title="Dashboard environment"
          tabs={[
            {
              label: 'Environment',
              language: 'bash',
              code: `# Point the dashboard at ${service.name}, then restart it\nexport ${assignment}`,
            },
          ]}
        />
      </div>
    </Card>
  )
}

export default ServiceNotConfigured
