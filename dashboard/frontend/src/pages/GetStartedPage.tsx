import { Rocket } from 'lucide-react'
import { EmptyState, IconTile, PageContainer, PageHeader } from '../ui'

export default function GetStartedPage() {
  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={Rocket} tone="blue" size="lg" />}
        title="Get started"
        description="Point any OpenAI-compatible client at the router and pick a public model name."
      />
      <EmptyState icon={Rocket} title="Connection guide" description="The endpoint, public model names and client examples appear here." />
    </PageContainer>
  )
}
