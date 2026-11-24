"""CLI del conversor UKJ."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .convert import convert_gguf
from .format import UkjError
from .inspect import format_summary, inspect_ukj, verify_ukj
from .subs import ensure_subs, extract_subs, verify_subs


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _default_output(source: str, density: str) -> str:
    stem, _ = os.path.splitext(source)
    return f"{stem}-{density}.ukj"


def cmd_convert(args: argparse.Namespace) -> int:
    out = args.output or _default_output(args.input, args.density)
    log = (lambda m: None) if args.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        result = convert_gguf(
            args.input,
            out,
            density=args.density,
            model_id=args.model_id,
            write_subs=not args.no_subs,
            subs_dir=args.subs_dir,
            force=args.force,
            continue_irregular=args.continue_irregular,
            log=log,
        )
    except (UkjError, ValueError, OSError) as exc:
        return _err(str(exc))
    if args.json:
        _print_json(result)
    else:
        print(f"OK · {result['output']}")
        print(f"model_id: {result['model_id']} · densidad: {result['density']}")
        print(f"payload sha256: {result['payload_sha256']}")
        if result.get("continued_tensors"):
            print(f"continue parameter: {result['continued_tensors']} tensores irregulares")
        if result.get("subs"):
            print(f".subs: {result['subs']}")
        print(f"tiempo: {result['seconds']} s")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        info = inspect_ukj(
            args.file,
            deep=args.deep,
            check_subs=args.subs,
            subs_dir=args.subs_dir,
        )
    except (UkjError, ValueError, OSError) as exc:
        return _err(str(exc))
    if args.json:
        _print_json(info)
    else:
        print(format_summary(info, show_tensors=args.tensors))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        result = verify_ukj(
            args.file,
            deep=not args.quick,
            check_subs=args.subs,
            subs_dir=args.subs_dir,
        )
    except (UkjError, ValueError, OSError) as exc:
        return _err(str(exc))
    if args.json:
        _print_json(result)
    else:
        print(f"OK · {result['path']}")
        print(f"model_id: {result['model_id']} · {result['tensor_count']} tensores")
        print(f"payload: {result['payload_bytes']} bytes")
        if args.quick:
            print("payload sha256: omitido (--quick)")
        if "subs" in result:
            print(f".subs: {result['subs']['status']} · {result['subs']['path']}")
    return 0 if result["ok"] else 1


def cmd_subs(args: argparse.Namespace) -> int:
    try:
        if args.action == "extract":
            path = extract_subs(
                args.file,
                subs_path=args.out,
                subs_dir=args.subs_dir,
                force=args.force,
                log=(lambda m: None) if args.quiet else (lambda m: print(m, file=sys.stderr)),
            )
            result = {"ok": True, "action": "extraído", "path": path}
        elif args.action == "verify":
            result = verify_subs(args.file, args.path, args.subs_dir)
        else:
            result = ensure_subs(
                args.file,
                subs_path=args.path,
                subs_dir=args.subs_dir,
                log=(lambda m: None) if args.quiet else (lambda m: print(m, file=sys.stderr)),
            )
    except (UkjError, ValueError, OSError) as exc:
        return _err(str(exc))
    if args.json:
        _print_json(result)
    else:
        print(f"{result.get('status', 'ok')} · {result.get('path', '')}")
        if result.get("message"):
            print(result["message"])
        if result.get("action"):
            print(f"acción: {result['action']}")
    return 0 if result.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ukj-convert",
        description="Convierte GGUF a UKJ v1 y gestiona sus .subs.",
    )
    p.add_argument("--version", action="version", version="ukj-convert 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="GGUF → .ukj (+ .subs)")
    c.add_argument("input", help="archivo .gguf")
    c.add_argument("-o", "--output", help="salida .ukj")
    c.add_argument(
        "-p",
        "--density",
        default="p1_p",
        help="cuantización p_p: p1_p…p8_p (p_p = p1_p)",
    )
    c.add_argument("--model-id", help="id del modelo (por defecto, nombre del GGUF)")
    c.add_argument("--no-subs", action="store_true", help="no materializar .subs")
    c.add_argument("--subs-dir", help="raíz de modelos de Cabrita (~/.cabrita/models)")
    c.add_argument("--force", action="store_true", help="sobrescribir la salida")
    c.add_argument(
        "--continue-irregular",
        action="store_true",
        help="reparte tensores irregulares en parámetros independientes (definición 46)",
    )
    c.add_argument("-q", "--quiet", action="store_true")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_convert)

    i = sub.add_parser("inspect", help="rutas, tensores, operadores, densidades y checksums")
    i.add_argument("file")
    i.add_argument("--json", action="store_true")
    i.add_argument("--tensors", action="store_true", help="listar todos los tensores")
    i.add_argument("--deep", action="store_true", help="recalcular el SHA-256 del payload")
    i.add_argument("--subs", action="store_true", help="comprobar el .subs asociado")
    i.add_argument("--subs-dir")
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("verify", help="validar el contenedor de principio a fin")
    v.add_argument("file")
    v.add_argument("--quick", action="store_true", help="no recalcular el payload")
    v.add_argument("--subs", action="store_true", help="comprobar también el .subs")
    v.add_argument("--subs-dir")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("subs", help="extraer, verificar o reparar .subs")
    s.add_argument("action", choices=["extract", "verify", "repair"])
    s.add_argument("file", help="archivo .ukj")
    s.add_argument("--out", help="ruta de salida para extract")
    s.add_argument("--path", help="ruta del .subs para verify/repair")
    s.add_argument("--subs-dir", help="raíz de modelos de Cabrita")
    s.add_argument("--force", action="store_true")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_subs)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
