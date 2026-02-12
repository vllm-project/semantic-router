#!/usr/bin/env python3
"""Generate RSA-2048 key pair and JWKS for Envoy jwt_authn filter.

Outputs:
  - jwks.json       : JWKS (public key) — mounted into Envoy container
  - private-key.pem : RSA private key — used to sign test JWTs

All files are written to the same directory as this script, unless --output-dir
is specified.
"""

import argparse
import base64
import json
import os
import sys
import uuid

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


def b64url(data: bytes) -> str:
    """Base64url-encode without padding (RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def int_to_b64url(n: int, length: int) -> str:
    """Encode a positive integer as base64url (big-endian, fixed length)."""
    return b64url(n.to_bytes(length, byteorder="big"))


def generate_rsa_keypair(key_size: int = 2048):
    """Generate an RSA key pair and return (private_key, public_key)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    public_key = private_key.public_key()
    return private_key, public_key


def build_jwks(public_key, kid: str) -> dict:
    """Build a JWKS dict from an RSA public key."""
    pub_numbers = public_key.public_numbers()
    # RSA 2048 → n is 256 bytes, e is 3 bytes (65537)
    n_bytes = (pub_numbers.n.bit_length() + 7) // 8
    e_bytes = (pub_numbers.e.bit_length() + 7) // 8
    return {
        "keys": [
            {
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "kid": kid,
                "n": int_to_b64url(pub_numbers.n, n_bytes),
                "e": int_to_b64url(pub_numbers.e, e_bytes),
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write output files (default: script directory)",
    )
    parser.add_argument(
        "--kid",
        default=None,
        help="Key ID for the JWKS key (default: random UUID)",
    )
    args = parser.parse_args()

    kid = args.kid or str(uuid.uuid4())
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating RSA-2048 key pair (kid={kid})...")
    private_key, public_key = generate_rsa_keypair()

    # Write private key PEM
    pem_path = os.path.join(args.output_dir, "private-key.pem")
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(pem_path, "wb") as f:
        f.write(pem_bytes)
    print(f"  Private key: {pem_path}")

    # Build and write JWKS
    jwks = build_jwks(public_key, kid)
    jwks_json = json.dumps(jwks, indent=2)
    jwks_path = os.path.join(args.output_dir, "jwks.json")
    with open(jwks_path, "w") as f:
        f.write(jwks_json + "\n")
    print(f"  JWKS:        {jwks_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
