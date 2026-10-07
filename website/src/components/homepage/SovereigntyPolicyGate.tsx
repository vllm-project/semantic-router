import React from 'react'
import Translate from '@docusaurus/Translate'
import { FiCheck, FiKey, FiLock, FiMapPin, FiShield } from 'react-icons/fi'
import styles from './SovereigntyPolicyGate.module.css'

type SovereigntyPolicyGateProps = {
  externalRequest: boolean
}

export default function SovereigntyPolicyGate({ externalRequest }: SovereigntyPolicyGateProps): React.JSX.Element {
  return (
    <div className={styles.gate} data-scope={externalRequest ? 'external' : 'private'}>
      <div className={styles.core} aria-hidden="true">
        <svg viewBox="0 0 96 96">
          <path className={styles.contacts} d="M 34 5 V 12 M 48 5 V 12 M 62 5 V 12 M 34 84 V 91 M 48 84 V 91 M 62 84 V 91 M 5 34 H 12 M 5 48 H 12 M 5 62 H 12 M 84 34 H 91 M 84 48 H 91 M 84 62 H 91" />
          <rect className={styles.board} x="12" y="12" width="72" height="72" rx="16" />
          <rect className={styles.innerBoard} x="18" y="18" width="60" height="60" rx="12" />
          <circle className={styles.scanTrack} cx="48" cy="48" r="32" />
          <circle className={styles.scan} cx="48" cy="48" r="32" />
          <path className={styles.shield} d="M 48 27 L 66 34 V 47 C 66 59 58 66 48 71 C 38 66 30 59 30 47 V 34 Z" />
          <path className={styles.check} d="M 39 48 L 46 55 L 58 41" />
          <circle className={styles.contactDot} cx="12" cy="48" r="2.5" />
          <circle className={styles.contactDot} cx="84" cy="48" r="2.5" />
        </svg>
      </div>

      <div className={styles.readout}>
        <span className={styles.label}>
          <span className={styles.statusDot} aria-hidden="true" />
          <Translate id="homepage.sovereigntyAI.gate">Policy gate</Translate>
        </span>
        <strong className={styles.verdict}>
          {externalRequest ? <FiCheck aria-hidden="true" /> : <FiLock aria-hidden="true" />}
          {externalRequest
            ? <Translate id="homepage.sovereigntyAI.externalApproved">External approved</Translate>
            : <Translate id="homepage.sovereigntyAI.keepOnPrem">Keep on-prem</Translate>}
        </strong>
        <span className={styles.decisionTrace} aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </div>

      <div className={styles.policies}>
        <span>
          <FiShield aria-hidden="true" />
          <Translate id="homepage.sovereigntyAI.gate.privacy">Privacy</Translate>
        </span>
        <span>
          <FiMapPin aria-hidden="true" />
          <Translate id="homepage.sovereigntyAI.gate.locality">Locality</Translate>
        </span>
        <span>
          <FiKey aria-hidden="true" />
          <Translate id="homepage.sovereigntyAI.gate.access">Access</Translate>
        </span>
      </div>
    </div>
  )
}
