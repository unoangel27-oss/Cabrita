//! Backend de inferencia Vulkan vía wgpu.
//!
//! El shader decuantiza los bloques GGML en la GPU y hace un producto punto
//! por fila con un workgroup de 256 hilos. Los pesos se suben una vez por
//! tensor y se cachean; la entrada y la salida reutilizan buffers.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;

use half::f16;

use crate::error::{Error, Result};
use crate::quant::QuantType;

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub max_storage_buffer: u64,
    pub max_buffer: u64,
}

struct WeightEntry {
    buffer: wgpu::Buffer,
    bytes: u64,
}

pub struct VulkanBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    weights: HashMap<String, WeightEntry>,
    weight_order: VecDeque<String>,
    weight_bytes: u64,
    max_weight_bytes: u64,
    x_buf: Option<wgpu::Buffer>,
    x_cap: usize,
    out_buf: Option<wgpu::Buffer>,
    out_cap: usize,
    stage_buf: Option<wgpu::Buffer>,
    stage_cap: usize,
    info: GpuInfo,
}

fn backend_name(backend: wgpu::Backend) -> String {
    format!("{backend:?}")
}

fn device_type_name(kind: wgpu::DeviceType) -> String {
    format!("{kind:?}")
}

fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: Default::default(),
    })
}

/// Consulta el adaptador Vulkan sin crear un device.
pub fn probe() -> Result<GpuInfo> {
    let instance = create_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| Error::new("no hay adaptador Vulkan disponible"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    Ok(GpuInfo {
        name: info.name,
        backend: backend_name(info.backend),
        device_type: device_type_name(info.device_type),
        driver: info.driver,
        driver_info: info.driver_info,
        max_storage_buffer: limits.max_storage_buffer_binding_size as u64,
        max_buffer: limits.max_buffer_size,
    })
}

impl VulkanBackend {
    pub fn new() -> Result<Self> {
        Self::with_cache_mb(4096)
    }

    pub fn with_cache_mb(cache_mb: u64) -> Result<Self> {
        let instance = create_instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| {
            Error::new("no hay adaptador Vulkan; comprueba `vulkaninfo` o usa --backend cpu")
        })?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cabrita-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
            },
            None,
        ))
        .map_err(|e| Error::new(format!("no se puede crear el device Vulkan: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cabrita-matvec"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan_matvec.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cabrita-matvec-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cabrita-matvec-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cabrita-matvec-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cabrita-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            uniform,
            weights: HashMap::new(),
            weight_order: VecDeque::new(),
            weight_bytes: 0,
            max_weight_bytes: cache_mb.saturating_mul(1024 * 1024),
            x_buf: None,
            x_cap: 0,
            out_buf: None,
            out_cap: 0,
            stage_buf: None,
            stage_cap: 0,
            info: GpuInfo {
                name: info.name,
                backend: backend_name(info.backend),
                device_type: device_type_name(info.device_type),
                driver: info.driver,
                driver_info: info.driver_info,
                max_storage_buffer: limits.max_storage_buffer_binding_size as u64,
                max_buffer: limits.max_buffer_size,
            },
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    pub fn cached_bytes(&self) -> u64 {
        self.weight_bytes
    }

    fn ensure_x(&mut self, len: usize) -> Result<()> {
        if self.x_cap < len {
            self.x_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cabrita-x"),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.x_cap = len;
        }
        Ok(())
    }

    fn ensure_out(&mut self, len: usize) -> Result<()> {
        let bytes = (len * 4) as u64;
        if self.out_cap < len {
            self.out_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cabrita-out"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.out_cap = len;
        }
        if self.stage_cap < len {
            self.stage_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cabrita-stage"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.stage_cap = len;
        }
        Ok(())
    }

    fn ensure_weight(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        if self.weights.contains_key(name) {
            return Ok(());
        }
        let padded = (bytes.len() + 3) & !3;
        if padded as u64 > self.info.max_buffer {
            return Err(Error::new(format!(
                "tensor '{name}' de {padded} bytes excede el máximo de buffer Vulkan ({})",
                self.info.max_buffer
            )));
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: padded as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytes);
        if padded > bytes.len() {
            self.queue.write_buffer(
                &buffer,
                bytes.len() as u64,
                &[0u8; 4][..padded - bytes.len()],
            );
        }

        // LRU simple: libera los tensores más antiguos si no cabe el nuevo.
        while self.weight_bytes + padded as u64 > self.max_weight_bytes {
            let Some(old) = self.weight_order.pop_front() else {
                break;
            };
            if let Some(entry) = self.weights.remove(&old) {
                self.weight_bytes = self.weight_bytes.saturating_sub(entry.bytes);
            }
        }
        self.weight_order.push_back(name.to_string());
        self.weight_bytes += padded as u64;
        self.weights.insert(
            name.to_string(),
            WeightEntry {
                buffer,
                bytes: padded as u64,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec(
        &mut self,
        name: &str,
        qt: QuantType,
        weights: &[u8],
        byte_offset: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Err(Error::new("Vulkan matvec: dimensiones vacías"));
        }
        if out.len() != rows || x.len() != cols {
            return Err(Error::new(
                "Vulkan matvec: dimensiones de entrada/salida inválidas",
            ));
        }
        let row_bytes = qt.row_bytes(cols)?;
        let needed = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::new("Vulkan matvec: tamaño desbordado"))?;
        if byte_offset + needed > weights.len() {
            return Err(Error::new(format!(
                "Vulkan matvec: '{name}' necesita {} bytes y hay {}",
                byte_offset + needed,
                weights.len()
            )));
        }
        if needed as u64 > self.info.max_storage_buffer {
            return Err(Error::new(format!(
                "tensor '{name}' de {needed} bytes excede el máximo de storage buffer Vulkan ({})",
                self.info.max_storage_buffer
            )));
        }

        self.ensure_weight(name, weights)?;
        self.ensure_x(cols)?;
        self.ensure_out(rows)?;

        let blocks = match qt {
            QuantType::Q2K | QuantType::Q3K | QuantType::Q6K => cols / crate::quant::QK_K,
            QuantType::F32 | QuantType::F16 => cols,
        };
        let qt_code = match qt {
            QuantType::F32 => 0u32,
            QuantType::F16 => 1u32,
            QuantType::Q2K => 10u32,
            QuantType::Q3K => 11u32,
            QuantType::Q6K => 14u32,
        };
        let params: [u32; 8] = [
            rows as u32,
            cols as u32,
            blocks as u32,
            qt_code,
            byte_offset as u32,
            row_bytes as u32,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&params));
        self.queue
            .write_buffer(self.x_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(x));
        let weight_buf = &self.weights.get(name).unwrap().buffer;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cabrita-matvec-bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.x_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.out_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cabrita-matvec"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cabrita-matvec-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            self.out_buf.as_ref().unwrap(),
            0,
            self.stage_buf.as_ref().unwrap(),
            0,
            (rows * 4) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.stage_buf.as_ref().unwrap().slice(..(rows * 4) as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(Error::new(format!(
                    "Vulkan: fallo al mapear la salida: {e}"
                )))
            }
            Err(e) => return Err(Error::new(format!("Vulkan: canal de lectura cerrado: {e}"))),
        }
        {
            let data = slice.get_mapped_range();
            let values: &[f32] = bytemuck::cast_slice(&data);
            out.copy_from_slice(&values[..rows]);
        }
        self.stage_buf.as_ref().unwrap().unmap();
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SelftestRow {
    pub quant: String,
    pub rows: usize,
    pub cols: usize,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub ok: bool,
}

fn lcg(seed: &mut u64, buf: &mut [u8]) {
    for b in buf.iter_mut() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (*seed >> 33) as u8;
    }
}

fn set_f16(bytes: &mut [u8], off: usize, value: f32) {
    bytes[off..off + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

/// Compara el shader con la ruta CPU para cada cuantización soportada.
pub fn selftest() -> Result<(GpuInfo, Vec<SelftestRow>)> {
    let mut backend = VulkanBackend::new()?;
    let mut rows = Vec::new();
    let cols = 512usize;
    let n_rows = 8usize;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.37).sin() * 1.5) + ((i % 7) as f32 * 0.03))
        .collect();

    for qt in [
        QuantType::F32,
        QuantType::F16,
        QuantType::Q2K,
        QuantType::Q3K,
        QuantType::Q6K,
    ] {
        let row_bytes = qt.row_bytes(cols)?;
        let mut bytes = vec![0u8; row_bytes * n_rows];
        lcg(&mut seed, &mut bytes);
        let bb = qt.block_bytes();
        if matches!(qt, QuantType::Q2K | QuantType::Q3K | QuantType::Q6K) {
            for r in 0..n_rows {
                let base = r * row_bytes;
                for b in 0..(cols / crate::quant::QK_K) {
                    let off = base + b * bb;
                    match qt {
                        QuantType::Q2K => {
                            set_f16(&mut bytes, off + 80, 0.5);
                            set_f16(&mut bytes, off + 82, 0.25);
                        }
                        QuantType::Q3K => set_f16(&mut bytes, off + 108, 0.5),
                        QuantType::Q6K => set_f16(&mut bytes, off + 208, 0.5),
                        _ => {}
                    }
                }
            }
        } else if qt == QuantType::F16 {
            for chunk in bytes.chunks_exact_mut(2) {
                let v = f16::from_f32(((chunk[0] as f32) - 128.0) / 128.0);
                chunk.copy_from_slice(&v.to_bits().to_le_bytes());
            }
        }

        let mut cpu = vec![0.0f32; n_rows];
        crate::quant::matvec(qt, &bytes, n_rows, cols, &x, &mut cpu)?;
        let mut gpu = vec![0.0f32; n_rows];
        let name = format!("selftest-{qt:?}");
        backend.matvec(&name, qt, &bytes, 0, n_rows, cols, &x, &mut gpu)?;

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (a, b) in cpu.iter().zip(&gpu) {
            let abs = (a - b).abs();
            max_abs = max_abs.max(abs);
            let denom = a.abs().max(1.0);
            max_rel = max_rel.max(abs / denom);
        }
        rows.push(SelftestRow {
            quant: format!("{qt:?}"),
            rows: n_rows,
            cols,
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            ok: max_rel < 2e-3,
        });
    }
    Ok((backend.info().clone(), rows))
}
