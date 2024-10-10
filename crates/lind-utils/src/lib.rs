#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};

#[allow(missing_docs)]
#[derive(Default)]
pub struct LindCageManager {
    cage_count: Mutex<i32>,
    condvar: Condvar,
}

impl LindCageManager {
    pub fn new(value: i32) -> Self {
        LindCageManager {
            cage_count: Mutex::new(value),
            condvar: Condvar::new(),
        }
    }

    pub fn increment(&self) {
        let mut cage_count = self.cage_count.lock().unwrap();
        *cage_count += 1;
    }

    pub fn decrement(&self) {
        let mut cage_count = self.cage_count.lock().unwrap();
        *cage_count -= 1;
        if *cage_count == 0 {
            self.condvar.notify_all();
        }
    }

    pub fn wait(&self) {
        let mut cage_count = self.cage_count.lock().unwrap();
        while *cage_count != 0 {
            cage_count = self.condvar.wait(cage_count).unwrap();
        }
    }
}

pub fn parse_env_var(env_var: &str) -> (String, Option<String>) {
    // Find the position of the first '=' character
    if let Some(pos) = env_var.find('=') {
        // If '=' is found, return the key and value as String and Some(String)
        let key = env_var[..pos].to_string();
        let value = env_var[pos + 1..].to_string();
        (key, Some(value))
    } else {
        // If '=' is not found, return the whole string as the key and None for the value
        (env_var.to_string(), None)
    }
}