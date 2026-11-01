#!/bin/sh
# Builds "Malachite.app" into build/ at the repo root (macOS only; ad-hoc signed, as the driver is).
#   sh updater/build.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
app="$here/../build/Malachite.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp "$here/Info.plist" "$app/Contents/"
xcrun swiftc -O -target x86_64-apple-macosx26.0 "$here"/*.swift -o "$app/Contents/MacOS/Malachite"
codesign --force --sign - "$app"
echo "$app"
