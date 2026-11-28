import { useLocation, useNavigate } from 'react-router-dom'
import { Compass, Search } from 'lucide-react'
import { Button, EmptyState, InlineCode, Kbd, PageContainer } from '../ui'
import { useShell } from '../shell/shell-context'

/** Unknown path inside the authenticated app: say so, then offer search and Home. */
export default function NotFoundPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const { setCommandOpen } = useShell()
  return (
    <PageContainer>
      <EmptyState
        icon={Compass}
        tone="slate"
        title="This page doesn't exist"
        description={
          <>
            Nothing lives at <InlineCode>{location.pathname}</InlineCode>. Search for the page you
            want or go back to Home.
          </>
        }
        actions={
          <>
            <Button variant="outline" onClick={() => setCommandOpen(true)}>
              <Search />
              Search pages
              <Kbd className="ml-1">⌘K</Kbd>
            </Button>
            <Button onClick={() => navigate('/')}>Go to Home</Button>
          </>
        }
      />
    </PageContainer>
  )
}
