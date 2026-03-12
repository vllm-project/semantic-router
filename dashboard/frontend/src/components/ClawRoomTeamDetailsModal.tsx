import { useEffect, useId } from 'react'
import { createPortal } from 'react-dom'

import styles from './ClawRoomChat.module.css'

interface TeamOption {
  id: string
  name: string
  role?: string
  vibe?: string
}

interface RoomOption {
  id: string
  name: string
}

interface MemberProfile {
  id: string
  displayName: string
  isLeader: boolean
  roleText: string
  vibeText: string
}

interface ClawRoomTeamDetailsModalProps {
  isOpen: boolean
  memberProfiles: MemberProfile[]
  onClose: () => void
  selectedRoom: RoomOption | null
  selectedTeam: TeamOption | null
  teamBriefText: string
}

const OPENCLAW_LOGO_SRC = '/openclaw.svg'

export default function ClawRoomTeamDetailsModal({
  isOpen,
  memberProfiles,
  onClose,
  selectedRoom,
  selectedTeam,
  teamBriefText,
}: ClawRoomTeamDetailsModalProps) {
  const titleId = useId()
  const memberSubtitle = `${memberProfiles.length} ${memberProfiles.length === 1 ? 'claw' : 'claws'}`

  useEffect(() => {
    if (!isOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  if (!isOpen || !selectedTeam || typeof document === 'undefined') {
    return null
  }

  return createPortal(
    <div className={styles.teamDetailsOverlay} onClick={onClose}>
      <div
        className={styles.teamDetailsDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="claw-room-team-details-dialog"
        onClick={event => event.stopPropagation()}
      >
        <div className={styles.teamDetailsHeader}>
          <div className={styles.teamDetailsHeaderCopy}>
            <span className={styles.teamDetailsEyebrow}>Team details</span>
            <h2 id={titleId} className={styles.teamDetailsTitle}>
              {selectedTeam.name}
            </h2>
          </div>
          <button
            type="button"
            className={styles.teamDetailsCloseButton}
            aria-label="Close team details"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className={styles.teamDetailsBody}>
          <section className={styles.teamDetailsHero}>
            {(selectedTeam.role || selectedTeam.vibe || selectedRoom) && (
              <div className={styles.metaInline}>
                {selectedTeam.role ? <span className={styles.metaPill}>{selectedTeam.role}</span> : null}
                {selectedTeam.vibe ? <span className={styles.metaPill}>{selectedTeam.vibe}</span> : null}
                {selectedRoom ? <span className={styles.metaPill}>Room · {selectedRoom.name}</span> : null}
              </div>
            )}
            <p className={styles.teamDetailsBrief}>{teamBriefText}</p>
            <div className={styles.teamDetailsSummaryGrid}>
              <div className={styles.metaCard}>
                <span className={styles.metaLabel}>Members</span>
                <span className={styles.metaValue}>{memberSubtitle}</span>
              </div>
              <div className={styles.metaCard}>
                <span className={styles.metaLabel}>Active room</span>
                <span className={styles.metaValue}>{selectedRoom?.name || 'No room selected'}</span>
              </div>
            </div>
          </section>

          <section className={styles.teamDetailsSection}>
            <div className={styles.teamDetailsSectionHeader}>
              <div>
                <div className={styles.roomsTitle}>Members</div>
                <div className={styles.roomsSubtitle}>{memberSubtitle}</div>
              </div>
            </div>

            {memberProfiles.length === 0 ? (
              <div className={styles.sidebarEmpty}>No workers in this team yet</div>
            ) : (
              <div className={styles.teamDetailsMemberGrid}>
                {memberProfiles.map(profile => (
                  <article
                    key={profile.id}
                    className={`${styles.memberResumeItem} ${profile.isLeader ? styles.memberResumeItemLeader : ''} ${styles.teamDetailsMemberCard}`}
                  >
                    <div className={styles.memberResumeHead}>
                      <span className={styles.memberResumeIdentity}>
                        <img
                          src={OPENCLAW_LOGO_SRC}
                          alt={profile.isLeader ? 'leader logo' : 'worker logo'}
                          className={styles.metaLogo}
                        />
                        <span className={styles.memberResumeName}>{profile.displayName}</span>
                      </span>
                      <span className={profile.isLeader ? styles.memberResumeRoleLeader : styles.memberResumeRoleWorker}>
                        {profile.isLeader ? 'LEADER' : 'WORKER'}
                      </span>
                    </div>
                    <div className={styles.memberResumeFactGrid}>
                      <div className={styles.memberResumeFact}>
                        <span className={styles.memberResumeFactLabel}>Role</span>
                        <span className={styles.memberResumeFactValue}>{profile.roleText}</span>
                      </div>
                      <div className={styles.memberResumeFact}>
                        <span className={styles.memberResumeFactLabel}>Style</span>
                        <span className={styles.memberResumeFactValue}>{profile.vibeText}</span>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>

        <div className={styles.teamDetailsFooter}>
          <button type="button" className={styles.teamDetailsFooterButton} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
