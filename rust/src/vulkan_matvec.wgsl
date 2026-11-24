// Matvec cuantizado en Vulkan (WGSL). Un workgroup de 256 hilos por fila:
// cada hilo decuantiza un elemento de cada bloque y la reducción suma el
// producto punto. Soporta F32, F16, Q2_K, Q3_K y Q6_K.

struct Params {
    rows: u32,
    cols: u32,
    blocks: u32,
    qt: u32,
    byte_offset: u32,
    row_bytes: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 256>;

fn byte_at(base: u32, i: u32) -> u32 {
    let pos = base + i;
    let word = w[pos / 4u];
    return (word >> ((pos % 4u) * 8u)) & 0xFFu;
}

fn u32_at(base: u32, i: u32) -> u32 {
    return byte_at(base, i)
        | (byte_at(base, i + 1u) << 8u)
        | (byte_at(base, i + 2u) << 16u)
        | (byte_at(base, i + 3u) << 24u);
}

fn f16_at(base: u32, i: u32) -> f32 {
    let bits = u32_at(base, i) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

fn f32_at(base: u32, i: u32) -> f32 {
    return bitcast<f32>(u32_at(base, i));
}

fn block_bytes(qt: u32) -> u32 {
    if (qt == 10u) { return 84u; }
    if (qt == 11u) { return 110u; }
    return 210u;
}

fn q2k(base: u32, j: u32) -> f32 {
    let chunk = j / 128u;
    let k = j % 128u;
    let shift = (k / 32u) * 2u;
    let half = (k % 32u) / 16u;
    let scale_idx = chunk * 8u + (k / 32u) * 2u + half;
    let q_idx = chunk * 32u + (k % 32u);
    let d = f16_at(base, 80u);
    let dmin = f16_at(base, 82u);
    let sc = byte_at(base, scale_idx);
    let q = (byte_at(base, 16u + q_idx) >> shift) & 3u;
    return d * f32(sc & 15u) * f32(q) - dmin * f32(sc >> 4u);
}

fn q3_scale(base: u32, idx: u32) -> f32 {
    let a0 = u32_at(base, 96u);
    let a1 = u32_at(base, 100u);
    let a2 = u32_at(base, 104u);
    let aux2 = ((a0 >> 4u) & 0x0f0f0f0fu) | (((a2 >> 4u) & 0x03030303u) << 4u);
    let aux3 = ((a1 >> 4u) & 0x0f0f0f0fu) | (((a2 >> 6u) & 0x03030303u) << 4u);
    let aux0 = (a0 & 0x0f0f0f0fu) | ((a2 & 0x03030303u) << 4u);
    let aux1 = (a1 & 0x0f0f0f0fu) | (((a2 >> 2u) & 0x03030303u) << 4u);
    var aux: array<u32, 4>;
    aux[0] = aux0;
    aux[1] = aux1;
    aux[2] = aux2;
    aux[3] = aux3;
    let b = (aux[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
    return f32(i32(b) - 32);
}

fn q3k(base: u32, j: u32) -> f32 {
    let chunk = j / 128u;
    let k = j % 128u;
    let shift = (k / 32u) * 2u;
    let idx = k % 32u;
    let is = chunk * 8u + (k / 32u) * 2u + (idx / 16u);
    let d = f16_at(base, 108u);
    var qv = i32((byte_at(base, 32u + chunk * 32u + idx) >> shift) & 3u);
    if ((byte_at(base, idx) & (1u << (chunk * 4u + k / 32u))) == 0u) {
        qv = qv - 4;
    }
    return d * q3_scale(base, is) * f32(qv);
}

fn q6_scale(base: u32, idx: u32) -> f32 {
    let b = byte_at(base, 192u + idx);
    let sv = i32(b) - select(0, 256, b >= 128u);
    return f32(sv);
}

fn q6k(base: u32, j: u32) -> f32 {
    let chunk = j / 128u;
    let k = j % 128u;
    let l = k % 32u;
    let quarter = k / 32u;
    let ql0 = byte_at(base, chunk * 64u + l);
    let ql1 = byte_at(base, chunk * 64u + l + 32u);
    let qh = byte_at(base, 128u + chunk * 32u + l);
    let d = f16_at(base, 208u);
    var q = 0u;
    var si = chunk * 8u + (l / 16u);
    if (quarter == 0u) {
        q = (ql0 & 0x0Fu) | ((qh & 3u) << 4u);
    } else if (quarter == 1u) {
        q = (ql1 & 0x0Fu) | (((qh >> 2u) & 3u) << 4u);
        si = si + 2u;
    } else if (quarter == 2u) {
        q = (ql0 >> 4u) | (((qh >> 4u) & 3u) << 4u);
        si = si + 4u;
    } else {
        q = (ql1 >> 4u) | (((qh >> 6u) & 3u) << 4u);
        si = si + 6u;
    }
    return d * q6_scale(base, si) * f32(i32(q) - 32);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let qt = p.qt;
    let row_base = p.byte_offset + row * p.row_bytes;
    var acc = 0.0;

    if (qt == 10u || qt == 11u || qt == 14u) {
        let bb = block_bytes(qt);
        for (var b = 0u; b < p.blocks; b = b + 1u) {
            let base = row_base + b * bb;
            var v = 0.0;
            if (qt == 10u) { v = q2k(base, tid); }
            else if (qt == 11u) { v = q3k(base, tid); }
            else { v = q6k(base, tid); }
            acc = acc + v * x[b * 256u + tid];
        }
    } else if (qt == 0u) {
        var c = tid;
        loop {
            if (c >= p.cols) { break; }
            acc = acc + f32_at(row_base, c * 4u) * x[c];
            c = c + 256u;
        }
    } else if (qt == 1u) {
        var c = tid;
        loop {
            if (c >= p.cols) { break; }
            acc = acc + f16_at(row_base, c * 2u) * x[c];
            c = c + 256u;
        }
    }

    partial[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        out[row] = partial[0];
    }
}
