import { Component, ErrorInfo, ReactNode } from 'react'
import { RotateCw } from 'lucide-react'

import { Alert, Button } from '../ui'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * Error Boundary - Catches React rendering errors and displays fallback UI
 * Prevents entire page from crashing due to component errors
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div role="alert" className="my-2">
          <Alert
            tone="danger"
            title="Something went wrong"
            actions={
              <Button type="button" variant="outline" size="xs" onClick={this.handleRetry}>
                <RotateCw />
                Try again
              </Button>
            }
          >
            <span className="[overflow-wrap:anywhere]">
              {this.state.error?.message || 'An unexpected error occurred'}
            </span>
          </Alert>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
