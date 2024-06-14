extern crate libc;
use libc::{printf, STDOUT_FILENO};
use rustposix::*;

fn main() {
    // lind_lindrustinit(0); // should be put to Store instantiate
    // let x = STDOUT_FILENO;
    // lind_rustposix_thread_init(1, 0);
    // let my_string = "Hello, world!\n";
    // let c_buf = my_string.as_ptr() as *const libc::c_void;
    // let buf_len = my_string.len();
    // lind_write(STDOUT_FILENO, c_buf, buf_len, 1); // fd_write
    // lind_lindrustfinalize();
}
