import { useEffect } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router-dom'

import { preloadDashboardRoute } from '../app/routeLoaders'
import ProductLoadingState from '../components/ProductLoadingState'
import { useAuth } from '../contexts/AuthContext'
import { sanitizeAuthTransitionTarget } from './authTransitionSupport'

export default function AuthTransitionPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { isAuthenticated, isLoading } = useAuth()
  const target = sanitizeAuthTransitionTarget(searchParams.get('to'), '/dashboard')

  useEffect(() => {
    void preloadDashboardRoute(target)
  }, [target])

  useEffect(() => {
    if (isAuthenticated && !isLoading) navigate(target, { replace: true })
  }, [isAuthenticated, isLoading, navigate, target])

  if (!isAuthenticated && !isLoading) {
    return <Navigate to="/login" replace state={{ from: target }} />
  }

  return <ProductLoadingState label="Opening dashboard" />
}
