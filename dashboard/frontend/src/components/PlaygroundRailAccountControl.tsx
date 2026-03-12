import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useAuth } from '../contexts/AuthContext'
import LayoutAccountControl from './LayoutAccountControl'

export default function PlaygroundRailAccountControl() {
  const [isOpen, setIsOpen] = useState(false)
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const accountName = user?.name?.trim() || 'Account'
  const accountEmail = user?.email?.trim() || 'Session pending'
  const accountPermissions = user?.permissions ?? []

  const handleClose = () => {
    setIsOpen(false)
  }

  const handleToggle = () => {
    setIsOpen(prev => !prev)
  }

  const handleLogout = () => {
    logout()
    setIsOpen(false)
    navigate('/login', { replace: true })
  }

  return (
    <LayoutAccountControl
      accountName={accountName}
      accountEmail={accountEmail}
      accountRole={user?.role}
      accountPermissions={accountPermissions}
      isOpen={isOpen}
      onToggle={handleToggle}
      onClose={handleClose}
      onLogout={handleLogout}
      variant="rail"
    />
  )
}
