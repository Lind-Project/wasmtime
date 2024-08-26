fn main() {
    println!("cargo:rustc-link-search=native=/home/lind-wasm/wasmtime/crates/rustposix");
    println!("cargo:rustc-link-lib=dylib=rustposix");
}
