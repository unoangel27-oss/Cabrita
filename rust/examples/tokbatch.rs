use cabrita::{tokenizer::Tokenizer, UkjFile};
use std::io::{self, BufRead};
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("ukj");
    let ukj = UkjFile::open(Path::new(&path), false, false).unwrap();
    let tok = Tokenizer::from_header(&ukj.header).unwrap();
    for line in io::stdin().lock().lines() {
        let text: String = serde_json::from_str(&line.unwrap()).unwrap();
        let ids = tok.encode(&text);
        println!("{}", serde_json::to_string(&ids).unwrap());
    }
}
