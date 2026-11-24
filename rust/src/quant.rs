//! Decuantización y productos matriz-vector para los tipos GGML que usa OLMoE.
//!
//! Los pesos se leen directamente del payload UKJ (mmap), sin copiarlos. Cada
//! bloque se decuantiza a una pila de 256 f32 y se multiplica por la entrada.

use half::f16;
use rayon::prelude::*;

use crate::error::{Error, Result};

pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    F32,
    F16,
    Q2K,
    Q3K,
    Q6K,
}

impl QuantType {
    pub fn from_ggml(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            14 => Ok(Self::Q6K),
            other => Err(Error::new(format!(
                "tipo GGML {other} no soportado por el motor CPU (F32, F16, Q2_K, Q3_K, Q6_K)"
            ))),
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q2K | Self::Q3K | Self::Q6K => QK_K,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q6K => 210,
        }
    }

    pub fn row_bytes(self, cols: usize) -> Result<usize> {
        let bs = self.block_size();
        if cols == 0 || cols % bs != 0 {
            return Err(Error::new(format!(
                "dimensión {cols} incompatible con bloques de {bs} del tipo {:?}",
                self
            )));
        }
        Ok((cols / bs) * self.block_bytes())
    }
}

fn f16_at(bytes: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[off], bytes[off + 1]])).to_f32()
}

fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// Decuantiza un bloque completo (256 valores) a `out`.
pub fn dequant_block(qt: QuantType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if out.len() < QK_K {
        return Err(Error::new("buffer de decuantización demasiado pequeño"));
    }
    match qt {
        QuantType::Q2K => {
            if bytes.len() < 84 {
                return Err(Error::new("bloque Q2_K truncado"));
            }
            let d = f16_at(bytes, 80);
            let dmin = f16_at(bytes, 82);
            let scales = &bytes[0..16];
            let qs = &bytes[16..80];
            let mut is = 0;
            let mut y = 0;
            for chunk in 0..2 {
                let q = &qs[chunk * 32..chunk * 32 + 32];
                let mut shift = 0;
                for _ in 0..4 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0f) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for &v in &q[..16] {
                        out[y] = dl * ((v >> shift) & 3) as f32 - ml;
                        y += 1;
                    }
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0f) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for &v in &q[16..] {
                        out[y] = dl * ((v >> shift) & 3) as f32 - ml;
                        y += 1;
                    }
                    shift += 2;
                }
            }
            Ok(())
        }
        QuantType::Q3K => {
            if bytes.len() < 110 {
                return Err(Error::new("bloque Q3_K truncado"));
            }
            let hmask = &bytes[0..32];
            let qs = &bytes[32..96];
            let raw_scales = &bytes[96..108];
            let d_all = f16_at(bytes, 108);

            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0f0f_0f0f;
            let mut aux = [0u32; 4];
            aux[0] = read_u32_le(raw_scales, 0);
            aux[1] = read_u32_le(raw_scales, 4);
            aux[2] = read_u32_le(raw_scales, 8);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let mut scales = [0i8; 16];
            for (i, slot) in scales.iter_mut().enumerate() {
                *slot = aux[i / 4].to_le_bytes()[i % 4] as i8;
            }

            let mut m = 1u8;
            let mut is = 0;
            let mut y = 0;
            for chunk in 0..2 {
                let q = &qs[chunk * 32..chunk * 32 + 32];
                let mut shift = 0;
                for _ in 0..4 {
                    for scale_index in 0..2 {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        for i in 0..16 {
                            let idx = i + 16 * scale_index;
                            let mut qv = ((q[idx] >> shift) & 3) as i8;
                            if hmask[idx] & m == 0 {
                                qv -= 4;
                            }
                            out[y] = dl * qv as f32;
                            y += 1;
                        }
                        is += 1;
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
            Ok(())
        }
        QuantType::Q6K => {
            if bytes.len() < 210 {
                return Err(Error::new("bloque Q6_K truncado"));
            }
            let ql = &bytes[0..128];
            let qh = &bytes[128..192];
            let scales = &bytes[192..208];
            let d = f16_at(bytes, 208);
            for chunk in 0..2 {
                let ql = &ql[chunk * 64..chunk * 64 + 64];
                let qh = &qh[chunk * 32..chunk * 32 + 32];
                let sc = &scales[chunk * 8..chunk * 8 + 8];
                let base = chunk * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    out[base + l] = d * (sc[is] as i8 as f32) * q1 as f32;
                    out[base + l + 32] = d * (sc[is + 2] as i8 as f32) * q2 as f32;
                    out[base + l + 64] = d * (sc[is + 4] as i8 as f32) * q3 as f32;
                    out[base + l + 96] = d * (sc[is + 6] as i8 as f32) * q4 as f32;
                }
            }
            Ok(())
        }
        _ => Err(Error::new("dequant_block solo acepta tipos cuantizados")),
    }
}

pub fn dequant_row(qt: QuantType, bytes: &[u8], cols: usize, out: &mut [f32]) -> Result<()> {
    match qt {
        QuantType::F32 => {
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        QuantType::F16 => {
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                out[i] = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
            Ok(())
        }
        _ => {
            let bb = qt.block_bytes();
            let blocks = cols / QK_K;
            for b in 0..blocks {
                dequant_block(qt, &bytes[b * bb..b * bb + bb], &mut out[b * QK_K..])?;
            }
            Ok(())
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < a.len() {
        acc0 += a[i] * b[i];
        i += 1;
    }
    (acc0 + acc1) + (acc2 + acc3)
}

/// `out[row] = dot(weights[row], x)` con `rows` filas de `cols` elementos.
pub fn matvec(
    qt: QuantType,
    weights: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let row_bytes = qt.row_bytes(cols)?;
    if weights.len() != rows * row_bytes {
        return Err(Error::new(format!(
            "pesos de {} bytes no coinciden con {rows}x{cols} ({})",
            weights.len(),
            rows * row_bytes
        )));
    }
    if x.len() != cols || out.len() != rows {
        return Err(Error::new(
            "matvec: dimensiones de entrada/salida inválidas",
        ));
    }
    match qt {
        QuantType::F32 | QuantType::F16 => {
            out.par_iter_mut().enumerate().for_each(|(r, value)| {
                let mut row = vec![0.0f32; cols];
                dequant_row(
                    qt,
                    &weights[r * row_bytes..(r + 1) * row_bytes],
                    cols,
                    &mut row,
                )
                .expect("fila F32/F16 válida");
                *value = dot(&row, x);
            });
        }
        _ => {
            let bb = qt.block_bytes();
            let blocks = cols / QK_K;
            out.par_iter_mut().enumerate().for_each(|(r, value)| {
                let mut acc = 0.0f32;
                let mut tmp = [0.0f32; QK_K];
                let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                for b in 0..blocks {
                    dequant_block(qt, &row[b * bb..b * bb + bb], &mut tmp)
                        .expect("bloque cuantizado válido");
                    acc += dot(&tmp, &x[b * QK_K..(b + 1) * QK_K]);
                }
                *value = acc;
            });
        }
    }
    Ok(())
}

/// `out[r * seq + t] = dot(weights[r], xs[t])`; el resultado queda por filas.
pub fn matmul(
    qt: QuantType,
    weights: &[u8],
    rows: usize,
    cols: usize,
    xs: &[f32],
    seq: usize,
    out: &mut [f32],
) -> Result<()> {
    let row_bytes = qt.row_bytes(cols)?;
    if weights.len() != rows * row_bytes {
        return Err(Error::new(format!(
            "pesos de {} bytes no coinciden con {rows}x{cols} ({})",
            weights.len(),
            rows * row_bytes
        )));
    }
    if xs.len() != seq * cols || out.len() != rows * seq {
        return Err(Error::new("matmul: dimensiones inválidas"));
    }
    match qt {
        QuantType::F32 | QuantType::F16 => {
            out.par_chunks_mut(seq).enumerate().for_each(|(r, dst)| {
                let mut row = vec![0.0f32; cols];
                dequant_row(
                    qt,
                    &weights[r * row_bytes..(r + 1) * row_bytes],
                    cols,
                    &mut row,
                )
                .expect("fila F32/F16 válida");
                for t in 0..seq {
                    dst[t] = dot(&row, &xs[t * cols..(t + 1) * cols]);
                }
            });
        }
        _ => {
            let bb = qt.block_bytes();
            let blocks = cols / QK_K;
            out.par_chunks_mut(seq).enumerate().for_each(|(r, dst)| {
                let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                let mut tmp = [0.0f32; QK_K];
                for t in 0..seq {
                    dst[t] = 0.0;
                }
                for b in 0..blocks {
                    dequant_block(qt, &row[b * bb..b * bb + bb], &mut tmp)
                        .expect("bloque cuantizado válido");
                    for t in 0..seq {
                        dst[t] += dot(&tmp, &xs[t * cols + b * QK_K..t * cols + (b + 1) * QK_K]);
                    }
                }
            });
        }
    }
    Ok(())
}

/// Convierte una salida por filas `[rows][seq]` a `[seq][rows]`.
pub fn rows_to_seq(src: &[f32], rows: usize, seq: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for r in 0..rows {
        for t in 0..seq {
            dst[t * rows + r] = src[r * seq + t];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q2k_block(d: f32, dmin: f32, scale: u8) -> Vec<u8> {
        let mut b = vec![0u8; 84];
        for s in &mut b[0..16] {
            *s = scale;
        }
        b[16..80].fill(0b01_01_01_01);
        b[80..82].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        b[82..84].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        b
    }

    #[test]
    fn q2k_dequant_simple() {
        let b = q2k_block(1.0, 0.5, 0x21);
        let mut out = [0.0f32; QK_K];
        dequant_block(QuantType::Q2K, &b, &mut out).unwrap();
        // scale = 1, min = 2, q = 1 -> 1*1 - 0.5*2 = 0
        for v in out {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn matvec_f32_matches_naive() {
        let rows = 4;
        let cols = 8;
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32) * 0.25 - 1.0;
        }
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = [0.5f32, -1.0, 2.0, 0.0, 1.5, 0.25, -0.75, 3.0];
        let mut out = vec![0.0f32; rows];
        matvec(QuantType::F32, &bytes, rows, cols, &x, &mut out).unwrap();
        for r in 0..rows {
            let expected: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            assert!((out[r] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn matmul_f32_matches_matvec() {
        let rows = 3;
        let cols = 8;
        let mut w = vec![0.0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7) % 11) as f32 * 0.1;
        }
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xs = [
            0.5f32, -1.0, 2.0, 0.0, 1.5, 0.25, -0.75, 3.0, 1.0, 0.0, -2.0, 0.5, 0.25, -1.0, 2.0,
            0.0,
        ];
        let mut out = vec![0.0f32; rows * 2];
        matmul(QuantType::F32, &bytes, rows, cols, &xs, 2, &mut out).unwrap();
        for t in 0..2 {
            for r in 0..rows {
                let expected: f32 = (0..cols).map(|c| w[r * cols + c] * xs[t * cols + c]).sum();
                assert!((out[r * 2 + t] - expected).abs() < 1e-4);
            }
        }
    }
}
