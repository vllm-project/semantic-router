import { Link } from 'react-router-dom'
import BrandLockup from './BrandLockup'
import styles from './PublicHeader.module.css'

const DOCS_URL = 'https://vllm-sr.ai/docs/intro/'
const GITHUB_URL = 'https://github.com/vllm-project/semantic-router'

export default function PublicHeader() {
  return (
    <header className={styles.header} data-testid="public-header">
      <div className={styles.inner} data-testid="public-header-content">
        <BrandLockup className={styles.brandPlacement} />

        <nav className={styles.nav} aria-label="Public navigation">
          <a
            className={styles.utilityLink}
            href={DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            Docs
          </a>
          <a
            className={styles.utilityLink}
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <Link className={styles.dashboardLink} to="/login">
            Enter Dashboard
          </Link>
        </nav>
      </div>
    </header>
  )
}
