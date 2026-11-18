import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { PageContainer } from '../ui'
import ChatComponent from '../components/ChatComponent'
import { isPlaygroundInvocation } from '../types/playgroundInvocation'
import { InsightsDataProviders } from './insightsModelLabelSupport'

const PlaygroundPage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const locationState =
    location.state && typeof location.state === 'object'
      ? (location.state as Record<string, unknown>)
      : null
  const invocation = isPlaygroundInvocation(locationState?.playgroundInvocation)
    ? locationState.playgroundInvocation
    : null
  const handleInvocationConsumed = useCallback(() => {
    navigate(`${location.pathname}${location.search}${location.hash}`, {
      replace: true,
      state: null,
    })
  }, [location.hash, location.pathname, location.search, navigate])

  return (
    <PageContainer width="full">
      <InsightsDataProviders>
        <ChatComponent
          endpoint="/api/router/v1/chat/completions"
          feedbackInsightsBasePath="/observe/requests"
          invocation={invocation}
          onInvocationConsumed={handleInvocationConsumed}
        />
      </InsightsDataProviders>
    </PageContainer>
  )
}

export default PlaygroundPage
