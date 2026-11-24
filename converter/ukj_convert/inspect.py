"""Inspección y verificación de un .ukj."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .container import open_ukj


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for u in units:
        if value < 1024 or u == units[-1]:
            return f"{value:.2f} {u}" if u != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def inspect_ukj(
    path: str,
    deep: bool = False,
    check_subs: bool = False,
    subs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    ukj = open_ukj(path, verify_payload=deep, verify_trailer=True)
    header = ukj.header
    model = header.get("model") or {}
    tensors = header["tensors"]
    routes = header["routes"]
    ops = header["operators"]
    densities = header.get("densities") or {}
    nsubs = sum(len(t["subs"]) for t in tensors)
    result: Dict[str, Any] = {
        "ok": True,
        "path": ukj.path,
        "format": header.get("format"),
        "version": header.get("version"),
        "model_id": model.get("id"),
        "model_name": model.get("name"),
        "architecture": model.get("architecture"),
        "quantization": model.get("quantization"),
        "parameter_count": model.get("parameter_count"),
        "tensor_count": len(tensors),
        "route_count": len(routes),
        "subparameter_count": nsubs,
        "default_density": densities.get("default"),
        "operators": [op.get("name") for op in ops],
        "payload_bytes": ukj.payload_len,
        "payload_sha256": ukj.payload_sha256,
        "header_sha256": ukj.header_sha256,
        "file_sha256": ukj.file_sha256,
        "size": os.path.getsize(ukj.path),
        "checks": {
            "header_sha256": True,
            "trailer": True,
            "payload_sha256": bool(deep),
        },
        "tensors": tensors,
        "routes": routes,
    }
    if check_subs:
        from .subs import verify_subs

        result["subs"] = verify_subs(ukj.path, subs_dir=subs_dir)
    return result


def verify_ukj(
    path: str,
    deep: bool = True,
    check_subs: bool = False,
    subs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    ukj = open_ukj(path, verify_payload=deep, verify_trailer=True)
    result: Dict[str, Any] = {
        "ok": True,
        "path": ukj.path,
        "model_id": ukj.header["model"]["id"],
        "tensor_count": len(ukj.header["tensors"]),
        "payload_bytes": ukj.payload_len,
        "checks": {
            "magic": True,
            "version": True,
            "header_json": True,
            "header_sha256": True,
            "tensor_index": True,
            "routes": True,
            "operators": True,
            "densities": True,
            "trailer": True,
            "payload_sha256": bool(deep),
        },
        "message": "contenedor UKJ válido",
    }
    if check_subs:
        from .subs import verify_subs

        result["subs"] = verify_subs(ukj.path, subs_dir=subs_dir)
        if not result["subs"]["ok"]:
            result["ok"] = False
            result["message"] = result["subs"]["message"]
    return result


def format_summary(info: Dict[str, Any], show_tensors: bool = False) -> str:
    lines = []
    lines.append(
        f"UKJ v{info['version']} · {info['model_id']} · "
        f"{info.get('quantization') or '?'} · {info['tensor_count']} tensores · "
        f"{info.get('parameter_count', 0):,} parámetros"
    )
    lines.append(
        f"payload {_fmt_bytes(info['payload_bytes'])} · "
        f"sha256 {info['payload_sha256']}"
    )
    lines.append(
        f"rutas: {info['route_count']} · operadores: "
        f"{', '.join(info['operators'])} · densidad: {info['default_density']} · "
        f"subparámetros: {info['subparameter_count']:,}"
    )
    if show_tensors:
        lines.append("")
        lines.append(
            f"{'tensor':44} {'tipo':6} {'dims':22} {'bytes':>12} "
            f"{'subs':>5} {'ruta':12} densidad"
        )
        for t in info["tensors"]:
            dims = "x".join(str(d) for d in t["dims"])
            lines.append(
                f"{t['name'][:44]:44} {t['type']:6} {dims[:22]:22} "
                f"{t['nbytes']:12,} {len(t['subs']):5} "
                f"{t.get('route', '')[:12]:12} {t.get('density', '')}"
            )
    if "subs" in info:
        s = info["subs"]
        lines.append(f".subs: {s['status']} · {s.get('path', '')}")
        if s.get("message"):
            lines.append(f"  {s['message']}")
    return "\n".join(lines)
