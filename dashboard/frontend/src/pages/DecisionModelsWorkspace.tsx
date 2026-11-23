import { lazy, Suspense } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Activity, ArrowUpRight, BrainCircuit, FlaskConical, Plus, Rocket } from 'lucide-react'
import {
  Button,
  IconTile,
  PageContainer,
  PageHeader,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '../ui'

const DecisionDeployTab = lazy(() => import('./DecisionDeployTab'))
const DecisionPlaygroundTab = lazy(() => import('./DecisionPlaygroundTab'))
const DecisionMonitoringTab = lazy(() => import('./DecisionMonitoringTab'))

const SYSTEM_ONE_DOCS = 'https://vllm-sr.ai/docs/system-one/overview'

type Tab = 'deploy' | 'playground' | 'monitoring'

function TabFallback() {
  return (
    <div className="flex flex-col gap-4" aria-busy="true">
      <Skeleton className="h-40 w-full rounded-xl" />
      <Skeleton className="h-64 w-full rounded-xl" />
    </div>
  )
}

function scrollToCatalog(attempt = 0) {
  const catalog = document.getElementById('decision-catalog')
  if (catalog) catalog.scrollIntoView({ behavior: 'smooth', block: 'start' })
  else if (attempt < 20) window.setTimeout(() => scrollToCatalog(attempt + 1), 60)
}

/** Decision models: deploy, try and monitor them, as tabs selected by `?tab=`. */
export default function DecisionModelsWorkspace() {
  const [params, setParams] = useSearchParams()
  const requested = params.get('tab')
  const tab: Tab = requested === 'playground' || requested === 'monitoring' ? requested : 'deploy'
  const select = (value: string) =>
    setParams(value === 'deploy' ? {} : { tab: value }, { replace: true })
  return (
    <PageContainer>
      <Tabs value={tab} onValueChange={select} className="gap-0">
        <PageHeader
          icon={<IconTile icon={BrainCircuit} tone="violet" size="lg" />}
          title="Decision models"
          description="Small System One models that answer typed routing questions for signals, decisions and the decision API."
          actions={
            <>
              <Button variant="outline" asChild>
                <a href={SYSTEM_ONE_DOCS} target="_blank" rel="noreferrer">
                  System One docs
                  <ArrowUpRight />
                </a>
              </Button>
              <Button
                onClick={() => {
                  if (tab !== 'deploy') select('deploy')
                  scrollToCatalog()
                }}
              >
                <Plus />
                Deploy a model
              </Button>
            </>
          }
        >
          <TabsList>
            <TabsTrigger value="deploy">
              <Rocket />
              Deploy
            </TabsTrigger>
            <TabsTrigger value="playground">
              <FlaskConical />
              Playground
            </TabsTrigger>
            <TabsTrigger value="monitoring">
              <Activity />
              Monitoring
            </TabsTrigger>
          </TabsList>
        </PageHeader>
        <Suspense fallback={<TabFallback />}>
          <TabsContent value="deploy">
            <DecisionDeployTab />
          </TabsContent>
          <TabsContent value="playground">
            <DecisionPlaygroundTab />
          </TabsContent>
          <TabsContent value="monitoring">
            <DecisionMonitoringTab />
          </TabsContent>
        </Suspense>
      </Tabs>
    </PageContainer>
  )
}
