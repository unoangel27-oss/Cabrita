use cabrita::tokenizer::Tokenizer;
use cabrita::UkjFile;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("ukj");
    let ukj = UkjFile::open(Path::new(&path), false, false).unwrap();
    let tok = Tokenizer::from_header(&ukj.header).unwrap();
    for text in std::env::args().skip(2) {
        let ids = tok.encode(&text);
        println!("{text:?} -> {ids:?}");
        println!("decoded: {:?}", tok.decode(&ids));
    }
}
