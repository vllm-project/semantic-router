#!/usr/bin/env bash
# ~27B M6 temporary node link, node B -> node A (workstation side; M5's link pattern). Node B's chain pulls node A's
# arm-seed relays and pushes mlx-diag collections and the IBX mixture; node A scores mlx-diag. The key lives only on
# node B in /data/dev2/tmp/27b-m6-xfer (mode 700, with known_hosts and the peer address); node A's authorized_keys line
# (comment dev2-27b-m6-xfer-temp) is `from=<node B's observed source>,command="rrsync /data/dev2/xfer/27b-m6",restrict`.
# Node addresses come from the private nodes.env and are never printed.
# Usage: m6-link.sh setup | check | remove
#   setup   key on node B; node A's host key scanned from node B and checked against node A's own; authorized_keys
#           backed up to authorized_keys.bak.27b-m6-<UTC>; a one-shot probe line learns node B's source address as
#           node A sees it, then is replaced by the restricted rrsync line
#   check   list the rrsync root from node B and round-trip a marker file
#   remove  delete node A's line (and the probe line if any) and node B's key directory
set -euo pipefail
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) B=$(grep '^node-b=' "$NODES" | cut -d= -f2-)
[ -n "$A" ] && [ -n "$B" ] || { echo "node-a / node-b missing in $NODES" >&2; exit 2; }
KEYDIR=/data/dev2/tmp/27b-m6-xfer COMMENT=dev2-27b-m6-xfer-temp ROOT=/data/dev2/xfer/27b-m6
AK=/root/.ssh/authorized_keys
SSHB="ssh -i $KEYDIR/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEYDIR/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
case "${1:?setup|check|remove}" in
  setup)
    if ssh -o BatchMode=yes "$A" "grep -q '$COMMENT' $AK"; then echo "node A already has a $COMMENT line" >&2; exit 66; fi
    ssh -o BatchMode=yes "$B" "set -e; umask 077; mkdir -p $KEYDIR; [ -f $KEYDIR/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -C $COMMENT -f $KEYDIR/id_ed25519; printf '%s\n' '${A#*@}' > $KEYDIR/peer; ssh-keyscan -t ed25519 \$(cat $KEYDIR/peer) 2> /dev/null > $KEYDIR/known_hosts; test -s $KEYDIR/known_hosts"
    scanned=$(ssh -o BatchMode=yes "$B" "cut -d' ' -f2,3 $KEYDIR/known_hosts")
    actual=$(ssh -o BatchMode=yes "$A" "cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub")
    [ "$scanned" = "$actual" ] || { echo "node A host key scanned from node B differs from node A's own" >&2; exit 3; }
    pub=$(ssh -o BatchMode=yes "$B" "cat $KEYDIR/id_ed25519.pub")
    [[ "$pub" == ssh-ed25519\ *\ $COMMENT ]] || { echo "unexpected public key format" >&2; exit 3; }
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    ssh -o BatchMode=yes "$A" "set -e; cp -p $AK $AK.bak.27b-m6-$stamp; mkdir -p $ROOT/relay $ROOT/mlx; printf '%s\n' 'command=\"echo \$SSH_CLIENT\",restrict $pub' >> $AK"
    src=$(ssh -o BatchMode=yes "$B" "$SSHB root@\$(cat $KEYDIR/peer) 2> /dev/null | cut -d' ' -f1") || true
    [[ "$src" =~ ^[0-9a-f.:]+$ ]] || {
      ssh -o BatchMode=yes "$A" "grep -v -F '$COMMENT' $AK > $AK.m6tmp && chmod 600 $AK.m6tmp && mv $AK.m6tmp $AK"
      echo "probe failed; node A's line was removed" >&2
      exit 3
    }
    ssh -o BatchMode=yes "$A" "set -e; grep -v -F '$COMMENT' $AK > $AK.m6tmp; printf '%s\n' 'from=\"$src\",command=\"/usr/bin/rrsync $ROOT\",restrict $pub' >> $AK.m6tmp; chmod 600 $AK.m6tmp; mv $AK.m6tmp $AK"
    echo "link set up (backup $AK.bak.27b-m6-$stamp on node A); run: m6-link.sh check" ;;
  check)
    ssh -o BatchMode=yes "$B" "set -e; $SSHB root@\$(cat $KEYDIR/peer) true 2> /dev/null && { echo 'shell access is NOT refused' >&2; exit 3; } || true; X='$SSHB'; rsync --list-only -e \"\$X\" root@\$(cat $KEYDIR/peer): > /dev/null; date -u +%FT%TZ > /data/dev2/tmp/m6-link-check.txt; rsync -a --mkpath -e \"\$X\" /data/dev2/tmp/m6-link-check.txt root@\$(cat $KEYDIR/peer):mlx/LINK-CHECK.txt; rm -f /data/dev2/tmp/m6-link-check.txt"
    ssh -o BatchMode=yes "$A" "test -s $ROOT/mlx/LINK-CHECK.txt && rm -f $ROOT/mlx/LINK-CHECK.txt && grep -c '$COMMENT' $AK"
    echo "link check passed (rrsync root listed, marker round-tripped, plain commands refused)" ;;
  remove)
    ssh -o BatchMode=yes "$A" "set -e; if grep -q -F '$COMMENT' $AK; then grep -v -F '$COMMENT' $AK > $AK.m6tmp; chmod 600 $AK.m6tmp; mv $AK.m6tmp $AK; fi; ! grep -q -F '$COMMENT' $AK"
    ssh -o BatchMode=yes "$B" "rm -rf $KEYDIR"
    echo "link removed (node A line gone, node B key directory deleted)" ;;
  *) echo "usage: m6-link.sh setup|check|remove" >&2; exit 2 ;;
esac
