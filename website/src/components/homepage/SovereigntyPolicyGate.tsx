import React from 'react'
import Translate from '@docusaurus/Translate'
import useBaseUrl from '@docusaurus/useBaseUrl'
import { FiCheck, FiKey, FiLock, FiMapPin, FiShield } from 'react-icons/fi'
import styles from './SovereigntyPolicyGate.module.css'

type SovereigntyPolicyGateProps = {
  externalRequest: boolean
}

export default function SovereigntyPolicyGate({ externalRequest }: SovereigntyPolicyGateProps): React.JSX.Element {
  const logo = useBaseUrl('/img/vllm-sr-mark.svg')

  return (
    <div className={styles.gate} data-scope={externalRequest ? 'external' : 'private'}>
      <div className={styles.core} aria-hidden="true">
        <span className={styles.brandMark}>
          <img src={logo} alt="" />
        </span>
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
