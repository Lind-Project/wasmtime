fn main() {
    println!("cargo:rustc-link-search=native=/wasmer/lib/rustposix");
    println!("cargo:rustc-link-lib=dylib=rustposix");
}
