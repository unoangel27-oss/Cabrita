use std::fs;

use serde_json::{json, Value};

use crate::reader::UkjFile;
use crate::subs::SubsStatus;

fn fmt_bytes(n: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = n as f64;
    for u in units {
        if value < 1024.0 || u == "TB" {
            return if u == "B" {
                format!("{n} B")
            } else {
                format!("{value:.2} {u}")
            };
        }
        value /= 1024.0;
    }
    format!("{n} B")
}

pub fn inspect_report(ukj: &UkjFile, deep: bool, subs: Option<&SubsStatus>) -> Value {
    let h = &ukj.header;
    let nsubs: usize = h.tensors.iter().map(|t| t.subs.len()).sum();
    let size = fs::metadata(ukj.path()).map(|m| m.len()).unwrap_or(0);
    let operators: Vec<&str> = h
        .operators
        .iter()
        .filter_map(|o| o.get("name").and_then(|v| v.as_str()))
        .collect();
    let mut report = json!({
        "ok": true,
        "path": ukj.path().display().to_string(),
        "format": h.format,
        "version": h.version,
        "model_id": h.model.id,
        "model_name": h.model.name,
        "architecture": h.model.architecture,
        "quantization": h.model.quantization,
        "parameter_count": h.model.parameter_count,
        "tensor_count": h.tensors.len(),
        "route_count": h.routes.len(),
        "subparameter_count": nsubs,
        "default_density": h.densities.default,
        "operators": operators,
        "payload_bytes": ukj.container.payload_len,
        "payload_sha256": ukj.container.payload_sha,
        "header_sha256": ukj.container.header_sha,
        "file_sha256": ukj.container.file_sha,
        "size": size,
        "checks": {
            "header_sha256": true,
            "trailer": true,
            "payload_sha256": deep,
        },
        "header": h,
    });
    if let Some(s) = subs {
        report["subs"] = serde_json::to_value(s).unwrap_or(Value::Null);
    }
    report
}

pub fn verify_report(ukj: &UkjFile, deep: bool, subs: Option<&SubsStatus>) -> Value {
    let ok = subs.map(|s| s.ok).unwrap_or(true);
    let mut report = json!({
        "ok": ok,
        "path": ukj.path().display().to_string(),
        "model_id": ukj.header.model.id,
        "tensor_count": ukj.header.tensors.len(),
        "payload_bytes": ukj.container.payload_len,
        "checks": {
            "magic": true,
            "version": true,
            "header_json": true,
            "header_sha256": true,
            "tensor_index": true,
            "routes": true,
            "operators": true,
            "densities": true,
            "trailer": true,
            "payload_sha256": deep,
        },
        "message": if ok { "contenedor UKJ válido" } else { "el .subs no es válido" },
    });
    if let Some(s) = subs {
        report["subs"] = serde_json::to_value(s).unwrap_or(Value::Null);
    }
    report
}

pub fn format_summary(report: &Value, show_tensors: bool) -> String {
    let mut out = String::new();
    let version = report.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
    let model = report
        .get("model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let quant = report
        .get("quantization")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let tensors = report
        .get("tensor_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let params = report
        .get("parameter_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    out.push_str(&format!(
        "UKJ v{version} · {model} · {quant} · {tensors} tensores · {params} parámetros\n"
    ));
    let payload = report
        .get("payload_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let sha = report
        .get("payload_sha256")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    out.push_str(&format!("payload {} · sha256 {sha}\n", fmt_bytes(payload)));
    let routes = report
        .get("route_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let density = report
        .get("default_density")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let nsubs = report
        .get("subparameter_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let ops = report
        .get("operators")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    out.push_str(&format!(
        "rutas: {routes} · operadores: {ops} · densidad: {density} · subparámetros: {nsubs}\n"
    ));
    let deep = report
        .get("checks")
        .and_then(|c| c.get("payload_sha256"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    out.push_str(&format!(
        "checks: cabecera OK · trailer OK · payload {}\n",
        if deep { "OK" } else { "omitido" }
    ));

    if show_tensors {
        if let Some(list) = report
            .get("header")
            .and_then(|h| h.get("tensors"))
            .and_then(|t| t.as_array())
        {
            out.push_str(&format!(
                "\n{:<44} {:<6} {:<22} {:>12} {:>5} {:<12} densidad\n",
                "tensor", "tipo", "dims", "bytes", "subs", "ruta"
            ));
            for t in list {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let ty = t.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                let dims = t
                    .get("dims")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64())
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join("x")
                    })
                    .unwrap_or_default();
                let nbytes = t.get("nbytes").and_then(|v| v.as_u64()).unwrap_or(0);
                let nsubs = t
                    .get("subs")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let route = t.get("route").and_then(|v| v.as_str()).unwrap_or("");
                let density = t.get("density").and_then(|v| v.as_str()).unwrap_or("");
                out.push_str(&format!(
                    "{:<44} {:<6} {:<22} {:>12} {:>5} {:<12} {}\n",
                    truncate(name, 44),
                    ty,
                    truncate(&dims, 22),
                    nbytes,
                    nsubs,
                    truncate(route, 12),
                    density
                ));
            }
        }
    }

    if let Some(subs) = report.get("subs").filter(|v| v.is_object()) {
        let status = subs.get("status").and_then(|v| v.as_str()).unwrap_or("?");
        let path = subs.get("path").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(&format!("\n.subs: {status} · {path}\n"));
        if let Some(msg) = subs.get("message").and_then(|v| v.as_str()) {
            out.push_str(&format!("  {msg}\n"));
        }
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n.saturating_sub(1)).collect::<String>() + "…"
    }
}
