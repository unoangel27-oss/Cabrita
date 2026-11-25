#!/usr/bin/env bash
# Construye una AppImage Linux autocontenida con Cabrita.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
APPDIR="$DIST/Cabrita.AppDir"
TOOLS="$ROOT/rust/target/tools"
VERSION="$(grep -m1 '^version' "$ROOT/rust/Cargo.toml" | cut -d'"' -f2)"

cd "$ROOT/rust"
cargo build --release

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/512x512/apps" "$APPDIR/usr/share/metainfo"

cp target/release/cabrita "$APPDIR/usr/bin/cabrita"
cp "$ROOT/docs/assets/logo-cabrita.png" "$APPDIR/cabrita.png"
cp "$ROOT/docs/assets/logo-cabrita.png" \
  "$APPDIR/usr/share/icons/hicolor/512x512/apps/cabrita.png"

cat > "$APPDIR/cabrita.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Cabrita
Comment=UKJ model runtime, chat and OpenAI-compatible server
Exec=cabrita
Icon=cabrita
Terminal=true
Categories=Development;
EOF
cp "$APPDIR/cabrita.desktop" "$APPDIR/usr/share/applications/cabrita.desktop"

cat > "$APPDIR/usr/share/metainfo/site.cabrita.appdata.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>site.cabrita</id>
  <metadata_license>MIT</metadata_license>
  <project_license>MIT</project_license>
  <name>Cabrita</name>
  <summary>Local UKJ model runtime</summary>
  <description><p>Native CPU/Vulkan inference, chat, world routing and an OpenAI-compatible HTTP API for UKJ models.</p></description>
  <launchable type="desktop-id">cabrita.desktop</launchable>
  <url type="homepage">https://github.com/unoangel27/cabrita</url>
  <developer><name>UKJ/Cabrita</name></developer>
  <content_rating type="oars-1.1"/>
  <provides><binary>cabrita</binary></provides>
</component>
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/cabrita" "$@"
EOF
chmod +x "$APPDIR/AppRun"

if ! command -v appimagetool >/dev/null 2>&1; then
  mkdir -p "$TOOLS"
  TOOL="$TOOLS/appimagetool-x86_64.AppImage"
  if [ ! -x "$TOOL" ]; then
    curl -L --retry 3 \
      -o "$TOOL" \
      "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$TOOL"
  fi
  APPIMAGETOOL="$TOOL"
else
  APPIMAGETOOL="$(command -v appimagetool)"
fi

OUT="$DIST/Cabrita-$VERSION-x86_64.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" "$APPDIR" "$OUT"
chmod +x "$OUT"
echo "AppImage: $OUT"
