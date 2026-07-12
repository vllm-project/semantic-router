import React from 'react'
import { Link } from 'react-router-dom'
import styles from './LayoutAccountControl.module.css'

interface LayoutAccountSecurityLinkProps {
  onSelect: () => void
}

const LayoutAccountSecurityLink: React.FC<LayoutAccountSecurityLinkProps> = ({ onSelect }) => (
  <Link className={styles.securityLink} to="/account/security" onClick={onSelect}>
    <span>Password &amp; security</span>
    <span aria-hidden="true">→</span>
  </Link>
)

export default LayoutAccountSecurityLink
