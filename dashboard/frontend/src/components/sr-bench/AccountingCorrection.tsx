import { Badge } from '../../ui'
import { money, number } from './model'
import type { Report } from './types'
import { Disclosure, Notice } from './benchUi'
import { codeBlockClass, factsClass, identityClass } from './benchStyles'

export default function AccountingCorrection({ report }: { report: Report | null }) {
  const correction = report?.provenance.accounting_correction
  if (!correction) return null
  return (
    <Notice
      tone={correction.qualified ? 'info' : 'warning'}
      as="aside"
      role="note"
      aria-label="Accounting correction"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4>Accounting correction recorded</h4>
        <Badge tone={correction.qualified ? 'success' : 'warning'} size="sm">
          {correction.qualified ? 'Accounting verified' : 'Partial accounting'}
        </Badge>
      </div>
      <p>
        Summary and comparison metrics use accounting reconciled from saved model responses.{' '}
        {correction.original_receipts_preserved
          ? 'Original call and case receipts remain unchanged.'
          : 'Receipt preservation is not confirmed.'}{' '}
        {correction.qualified
          ? ''
          : 'Unverifiable usage remains unknown; known spend is not a complete bill.'}
      </p>
      <dl className={factsClass}>
        <div>
          <dt>Corrected calls</dt>
          <dd>{number(correction.corrected_call_count)}</dd>
        </div>
        <div>
          <dt>Verified / unverifiable</dt>
          <dd>
            {number(correction.verified_call_count)} / {number(correction.unverifiable_call_count)}
          </dd>
        </div>
        <div>
          <dt>Original known spend</dt>
          <dd>{money(correction.original_known_spend_usd)}</dd>
        </div>
        <div>
          <dt>Reconciled known spend</dt>
          <dd>{money(correction.corrected_known_spend_usd)}</dd>
        </div>
        <div>
          <dt>New model requests</dt>
          <dd>{number(correction.model_requests)}</dd>
        </div>
      </dl>
      <Disclosure summary="Accounting correction receipt" bare className="my-0">
        <dl className={identityClass}>
          <dt>Version</dt>
          <dd>{correction.version}</dd>
          <dt>Recorded</dt>
          <dd>
            <time dateTime={correction.created_at}>{correction.created_at}</time>
          </dd>
          <dt>Evidence SHA-256</dt>
          <dd>
            <code>{correction.evidence_sha256}</code>
          </dd>
        </dl>
        <pre className={codeBlockClass}>{JSON.stringify(correction, null, 2)}</pre>
      </Disclosure>
    </Notice>
  )
}
