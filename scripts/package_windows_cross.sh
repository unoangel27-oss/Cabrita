#!/usr/bin/env bash
# Cross-compila `dist/cabrita.exe` desde Linux sin privilegios.
#
# Descarga los .deb de mingw-w64 (binutils, gcc y cabeceras), los extrae en un
# prefijo de caché y compila el target `x86_64-pc-windows-gnu` de Rust. No toca
# el sistema y no necesita sudo ni Wine.
#
# Uso:
#   bash scripts/package_windows_cross.sh
#   MINGW_PREFIX=/tmp/opencode/mingw bash scripts/package_windows_cross.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${MINGW_PREFIX:-$HOME/.cache/cabrita-mingw}"
DEBS="$PREFIX/debs"
SYSROOT="$PREFIX/root"
TRIPLE="x86_64-w64-mingw32"
RUST_TARGET="x86_64-pc-windows-gnu"
GCC="$SYSROOT/usr/bin/$TRIPLE-gcc-posix"

need_debs() {
  [ -x "$GCC" ] && [ -d "$SYSROOT/usr/share/mingw-w64/include" ] && return 1
  return 0
}

if need_debs; then
  mkdir -p "$DEBS"
  cd "$DEBS"
  apt-get download \
    gcc-mingw-w64-x86-64-posix \
    gcc-mingw-w64-base \
    binutils-mingw-w64-x86-64 \
    mingw-w64-x86-64-dev \
    mingw-w64-common
  for deb in ./*.deb; do
    dpkg-deb -x "$deb" "$SYSROOT"
  done
fi

REAL_GCC="$SYSROOT/usr/$TRIPLE"
CATALOG="$PREFIX/bin"
mkdir -p "$CATALOG"
cat > "$CATALOG/$TRIPLE-gcc" <<EOF
#!/bin/sh
exec "$GCC" -B"$REAL_GCC" -isystem "$REAL_GCC/include" -L"$REAL_GCC/lib" "\$@"
EOF
chmod +x "$CATALOG/$TRIPLE-gcc"

rustup target add "$RUST_TARGET" >/dev/null 2>&1 || true

export PATH="$SYSROOT/usr/bin:$CATALOG:$PATH"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$CATALOG/$TRIPLE-gcc"
export CC_x86_64_pc_windows_gnu="$CATALOG/$TRIPLE-gcc"
export AR_x86_64_pc_windows_gnu="$SYSROOT/usr/bin/$TRIPLE-ar"

cd "$ROOT/rust"
cargo build --release --target "$RUST_TARGET" --bin cabrita

mkdir -p "$ROOT/dist"
cp "target/$RUST_TARGET/release/cabrita.exe" "$ROOT/dist/cabrita.exe"
echo "Windows: $ROOT/dist/cabrita.exe"
