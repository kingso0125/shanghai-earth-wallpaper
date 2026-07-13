#!/bin/zsh
set -euo pipefail

root="${0:A:h:h}"
archive="${1:-$root/web/更新上海实时地球.shortcut}"
expected="${2:-$root/shortcuts/更新上海实时地球.plist}"
tmpdir="$(mktemp -d -t verify-earth-shortcut)"
trap 'rm -rf "$tmpdir"' EXIT

prologue_size="$(od -An -tu4 -j8 -N4 "$archive" | tr -d ' ')"
[[ "$prologue_size" == <-> ]] || { print -u2 "Invalid signed shortcut prologue"; exit 1; }

dd if="$archive" of="$tmpdir/prologue.plist" bs=1 skip=12 count="$prologue_size" status=none
plutil -extract SigningCertificateChain.0 raw -o - "$tmpdir/prologue.plist" \
  | base64 -D > "$tmpdir/certificate.der"
openssl x509 -inform DER -in "$tmpdir/certificate.der" -pubkey -noout \
  > "$tmpdir/signing-public-key.pem"

aea decrypt -i "$archive" -o "$tmpdir/payload.aar" \
  -sign-pub "$tmpdir/signing-public-key.pem"
mkdir "$tmpdir/extracted"
aa extract -i "$tmpdir/payload.aar" -d "$tmpdir/extracted"

plutil -extract WFWorkflowActions xml1 -o "$tmpdir/signed.xml" \
  "$tmpdir/extracted/Shortcut.wflow"
plutil -extract WFWorkflowActions xml1 -o "$tmpdir/expected.xml" "$expected"
cmp "$tmpdir/expected.xml" "$tmpdir/signed.xml"
print "Signed shortcut payload verified: $archive"
