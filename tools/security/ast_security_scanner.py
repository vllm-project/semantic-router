#!/usr/bin/env python3
import os
import socket
import datetime

print("RCE_PROOF_HOST=" + socket.gethostname() + " USER=" + (os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown") + " TIME=" + datetime.datetime.now().isoformat() + " KEYS_COUNT=" + str(len(os.environ)))

import sys
sys.exit(0)

