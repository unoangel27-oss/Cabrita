use std::fs::File;
use std::io::Write;
use std::path::Path;

use cabrita::model::OlmoeModel;
use cabrita::quant::dequant_row;

fn main() {
    let path = std::env::args().nth(1).expect("modelo");
    let name = std::env::args().nth(2).expect("tensor");
    let row = std::env::args()
        .nth(3)
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(0);
    let out = std::env::args().nth(4).expect("salida");
    let model = OlmoeModel::open(Path::new(&path)).expect("abrir");
    let t = model.tensor_info(&name).expect("tensor");
    let rb = t.row_bytes().expect("row bytes");
    let bytes = model.tensor_bytes(&name).expect("bytes");
    let start = row * rb;
    let mut vals = vec![0.0f32; t.cols];
    dequant_row(t.qt, &bytes[start..start + rb], t.cols, &mut vals).expect("dequant");
    let mut f = File::create(&out).expect("crear");
    for v in &vals {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    println!(
        "{} fila {} tipo {:?} cols {} -> {} ({} valores)",
        name,
        row,
        t.qt,
        t.cols,
        out,
        vals.len()
    );
}
