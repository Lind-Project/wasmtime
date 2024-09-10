#![allow(dead_code)]

use anyhow::{anyhow, Result};
use rustposix::{lind_exit, lind_fork, LindCageManager};
use wasi_common::WasiCtx;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use wasmtime::{AsContext, AsContextMut, Caller, ExternType, Linker, Module, SharedMemory, Store, Val, Extern, OnCalledAction, RewindingReturn, StoreOpaque, InstanceId};

use wasmtime_environ::MemoryIndex;

const ASYNCIFY_START_UNWIND: &str = "asyncify_start_unwind";
const ASYNCIFY_STOP_UNWIND: &str = "asyncify_stop_unwind";
const ASYNCIFY_START_REWIND: &str = "asyncify_start_rewind";
const ASYNCIFY_STOP_REWIND: &str = "asyncify_stop_rewind";

#[derive(Clone)]
pub struct WasiForkCtx<T, U> {
    // instance_pre seems to be a static data and won't be modified anymore once created
    // so we may not need to make a deep clone of it
    // instance_pre: Arc<InstancePre<T>>,
    linker: Linker<T>,
    module: Module,

    // cageid is stored at preview1_ctx, so we do not need a duplicated field here
    // cageid: u64,

    // current pid associated with this ctx
    pid: i32,
    // next_pid and next_cageid should be shared between processes
    // next_pid: Arc<AtomicI32>,
    next_cageid: Arc<AtomicU64>,

    lind_manager: Arc<LindCageManager>,

    run_command: U
}

impl<T: Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync> WasiForkCtx<T, U> {
    pub fn new(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U) -> Result<Self> {
        // this method should only be called once from run.rs, other instances of WasiForkCtx
        // are supposed to be created from fork() method
        // let instance_pre = Arc::new(linker.instantiate_pre(&module)?);
        let pid = 1;
        // let next_pid = Arc::new(AtomicI32::new(0));
        let next_cageid = Arc::new(AtomicU64::new(1)); // cageid starts from 1
        Ok(Self { linker, module: module.clone(), pid, next_cageid, lind_manager: lind_manager.clone(), run_command })
    }

    pub fn new_with_pid(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U, pid: i32, next_cageid: Arc<AtomicU64>) -> Result<Self> {
        Ok(Self { linker, module: module.clone(), pid, next_cageid, lind_manager: lind_manager.clone(), run_command })
    }

    pub fn fork_call(&self, mut caller: &mut Caller<'_, T>,
                get_cx: impl Fn(&mut T) -> &mut WasiForkCtx<T, U> + Send + Sync + Copy + 'static,
                get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
                fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static) -> Result<i32> {
        // if !support_asyncify(instance_pre.module()) {
        //     log::error!("failed to find asyncify functions");
        //     return Ok(-1);
        // }
        // if !has_correct_signature(instance_pre.module()) {
        //     log::error!("the exported asyncify functions have incorrect signatures");
        //     return Ok(-1);
        // }

        // if fork is called during the rewinding process
        // that would mean fork has completed and we want to stop the rewind
        // and return the fork result
        if caller.as_context().get_rewinding_state().rewinding {
            if let Some(asyncify_stop_rewind_extern) = caller.get_export(ASYNCIFY_STOP_REWIND) {
                match asyncify_stop_rewind_extern {
                    Extern::Func(asyncify_stop_rewind) => {
                        match asyncify_stop_rewind.typed::<(), ()>(&caller) {
                            Ok(func) => {
                                let _res = func.call(&mut caller, ());
                            }
                            Err(err) => {
                                println!("the signature of asyncify_stop_rewind is not correct: {:?}", err);
                                return Ok(-1);
                            }
                        }
                    },
                    _ => {
                        println!("asyncify_stop_rewind export is not a function");
                        return Ok(-1);
                    }
                }
            }
            else {
                println!("asyncify_stop_rewind export not found");
                return Ok(-1);
            }

            let retval = caller.as_context().get_rewinding_state().retval;

            caller.as_context_mut().set_rewinding_state(RewindingReturn {
                rewinding: false,
                retval: 0,
            });

            return Ok(retval);
        }

        // get the base address of the memory
        let handle = caller.as_context().0.instance(InstanceId::from_index(0));
        let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
        let address = defined_memory.base;

        // get the stack pointer global
        let stack_pointer;
        if let Some(sp_extern) = caller.get_export("__stack_pointer") {
            match sp_extern {
                Extern::Global(sp) => {
                    match sp.get(&mut caller) {
                        Val::I32(val) => {
                            stack_pointer = val;
                        }
                        _ => {
                            println!("__stack_pointer export is not an i32");
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("__stack_pointer export is not a Global");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("__stack_pointer export not found");
            return Ok(-1);
        }

        // start unwind
        if let Some(asyncify_start_unwind_extern) = caller.get_export(ASYNCIFY_START_UNWIND) {
            match asyncify_start_unwind_extern {
                Extern::Func(asyncify_start_unwind) => {
                    match asyncify_start_unwind.typed::<i32, ()>(&caller) {
                        Ok(func) => {
                            let unwind_pointer: u64 = 0;
                            // 8 because we need to store unwind_data_start and unwind_data_end
                            // at the beginning of the unwind stack as the parameter for asyncify_start_unwind
                            // each of them are u64, so together is 8 bytes
                            let unwind_data_start: u64 = unwind_pointer + 8;
                            let unwind_data_end: u64 = stack_pointer as u64;
    
                            unsafe {
                                *(address as *mut u64) = unwind_data_start;
                                *(address as *mut u64).add(1) = unwind_data_end;
                            }
    
                            let _res = func.call(&mut caller, unwind_pointer as i32);
                        }
                        Err(err) => {
                            println!("the signature of asyncify_start_unwind function is not correct: {:?}", err);
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("asyncify_start_unwind export is not a function");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("asyncify_start_unwind export not found");
            return Ok(-1);
        }

        // get the asyncify_stop_unwind and asyncify_start_rewind, which will later
        // be used when the unwind process finished
        let asyncify_stop_unwind_func;
        let asyncify_start_rewind_func;

        if let Some(asyncify_stop_unwind_extern) = caller.get_export(ASYNCIFY_STOP_UNWIND) {
            match asyncify_stop_unwind_extern {
                Extern::Func(asyncify_stop_unwind) => {
                    match asyncify_stop_unwind.typed::<(), ()>(&caller) {
                        Ok(func) => {
                            asyncify_stop_unwind_func = func;
                        }
                        Err(err) => {
                            println!("the signature of asyncify_stop_unwind function is not correct: {:?}", err);
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("asyncify_stop_unwind export is not a function");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("asyncify_stop_unwind export not found");
            return Ok(-1);
        }

        if let Some(asyncify_start_rewind_extern) = caller.get_export(ASYNCIFY_START_REWIND) {
            match asyncify_start_rewind_extern {
                Extern::Func(asyncify_start_rewind) => {
                    match asyncify_start_rewind.typed::<i32, ()>(&caller) {
                        Ok(func) => {
                            asyncify_start_rewind_func = func;
                        }
                        Err(err) => {
                            println!("the signature of asyncify_start_rewind function is not correct: {:?}", err);
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("asyncify_start_rewind export is not a function");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("asyncify_start_rewind export not found");
            return Ok(-1);
        }

        let cloned_address = address as u64;

        // ready for creating the child instance
        // let instance_pre = Arc::new(self.linker.instantiate_pre(&self.module)?);
        // let instance_pre = self.instance_pre.clone();

        // let module = instance_pre.module();
        // let offsets = module.offsets();
        // let vm_offset = offsets.vmctx_imported_memories_begin();
        // println!("vmoffsets: {:?}", offsets);

        let mut child_host = fork_host(caller.data());
        let child_cageid = self.next_cage_id();
        if let None = child_cageid {
            println!("running out of cageid!");
            // return Ok(-1);
        }
        let child_cageid = child_cageid.unwrap();

        // set up child wasi_ctx cage id
        let child_wasi_ctx = get_wasi_cx(&mut child_host);
        let parent_cageid = child_wasi_ctx.get_lind_cageid();
        // let child_cageid = parent_cageid + 1;
        child_wasi_ctx.set_lind_cageid(child_cageid);

        let engine = self.module.engine().clone();

        // set up unwind callback function
        let store = caller.as_context_mut().0;
        let is_parent_thread = store.is_thread();
        store.set_on_called(Box::new(move |mut store| {
            // let unwind_stack_finish;

            // let address = cloned_address as *mut u64;
            // let unwind_start_address = (cloned_address + 8) as *mut u64;

            // unsafe {
            //     unwind_stack_finish = *address;
            // }

            // let unwind_size = unwind_stack_finish - 8;
            // let mut unwind_stack = Vec::with_capacity(unwind_size as usize);

            // unsafe {
            //     let src_slice = std::slice::from_raw_parts(unwind_start_address as *mut u8, unwind_size as usize);
            //     unwind_stack.extend_from_slice(src_slice);
            // }

            // unwind finished and we need to stop the unwind
            let _res = asyncify_stop_unwind_func.call(&mut store, ());

            let rewind_pointer: u64 = 0;

            // use a barrier to make sure the child has fully copied parent's memory before parent
            // resumes its execution
            let barrier = Arc::new(Barrier::new(2));
            let barrier_clone = Arc::clone(&barrier);

            let builder = thread::Builder::new().name(format!("lind-fork-{}", child_cageid));
            builder.spawn(move || {
                // let mut child_host = caller.data().clone();
                // let mut child_ctx = get_cx(&child_host);

                // create a new instance
                // let mut store = Store::new(&engine, child_host);
                let store_inner = Store::<T>::new_inner(&engine);

                let child_ctx = get_cx(&mut child_host);
                child_ctx.pid = child_cageid as i32;

                child_ctx.fork_memory(&store_inner);
                let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());
                // println!("instantiate");

                let lind_manager = child_ctx.lind_manager.clone();
                let mut store = Store::new_with_inner(&engine, child_host, store_inner);

                // if parent is a thread, so does the child
                if is_parent_thread {
                    store.set_is_thread(true);
                }

                let instance = instance_pre.instantiate(&mut store).unwrap();

                // copy the entire memory from parent, note that the unwind data is also copied together
                // with the memory
                let child_address: *mut u8;
                let address_length: usize;

                // get the base address of the memory
                {
                    let handle = store.inner_mut().instance(InstanceId::from_index(0));
                    let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
                    child_address = defined_memory.base;
                    address_length = defined_memory.current_length.load(Ordering::SeqCst);
                }

                if child_address == 0 as *mut u8 {
                    println!("no memory found for child");
                    return -1;
                }

                // println!("cloned_address: {:?}, child_address: {:?}", cloned_address as *mut u8, child_address);

                unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, child_address, address_length); }

                // new cage created, increment the cage counter
                lind_manager.increment();
                // create the cage in rustposix via rustposix fork
                lind_fork(parent_cageid, child_cageid);

                barrier_clone.wait();

                // get the asyncify_rewind_start and module start function
                let child_rewind_start;

                match instance.get_typed_func::<i32, ()>(&mut store, ASYNCIFY_START_REWIND) {
                    Ok(func) => {
                        child_rewind_start = func;
                    },
                    Err(_error) => {
                        return -1;
                    }
                };

                // rewind the child
                let _ = child_rewind_start.call(&mut store, rewind_pointer as i32);

                // set up rewind state and fork return value for child
                store.as_context_mut().set_rewinding_state(RewindingReturn {
                    rewinding: true,
                    retval: 0,
                });

                if store.is_thread() {
                    // fork inside a thread is possible but not common
                    // when fork happened inside a thread, it will only fork that specific thread
                    // and left other threads un-copied.
                    // to support this, we can just store the thread start args and calling wasi_thread_start
                    // with the same start args here instead of _start entry.
                    // however, since this is not a common practice, so we do not support this right now
                    return -1;
                } else {
                    let child_start_func = instance
                        .get_func(&mut store, "_start")
                        .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();

                    let ty = child_start_func.ty(&store);

                    let values = Vec::new();
                    let mut results = vec![Val::null_func_ref(); ty.results().len()];

                    let _invoke_res = child_start_func
                        .call(&mut store, &values, &mut results);

                    let exit_code = results.get(0).expect("_start function does not have a return value");
                    match exit_code {
                        Val::I32(val) => {
                            // exit the cage
                            lind_exit(child_cageid, *val);
                        },
                        _ => {
                            println!("unexpected _start function return type!");
                        }
                    }

                    // the cage just exited, decrement the cage counter
                    lind_manager.decrement();
                }

                return 0;
            }).unwrap();

            barrier.wait();

            let _ = asyncify_start_rewind_func.call(&mut store, rewind_pointer as i32);

            // set up rewind state and fork return value for parent
            store.set_rewinding_state(RewindingReturn {
                rewinding: true,
                retval: child_cageid as i32,
            });

            // return InvokeAgain here would make parent re-invoke main
            return Ok(OnCalledAction::InvokeAgain);
        }));

        return Ok(0);
    }

    pub fn execv_call(&self, mut caller: &mut Caller<'_, T>,
                             path: i64,
                             argv: i64,
                             exec: impl Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>) -> Result<Vec<Val>> + Send + Sync + Copy + 'static,
                     ) -> Result<i32> {
        // get the base address of the memory
        let handle = caller.as_context().0.instance(InstanceId::from_index(0));
        let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
        let address = defined_memory.base;

        let path_ptr = ((address as i64) + path) as *const u8;
        let path_str;

        // NOTE: the address passed from wasm module is 32-bit address
        let argv_ptr = ((address as i64) + argv) as *const *const u8;
        let mut args = Vec::new();

        unsafe {
            // Manually find the null terminator
            let mut len = 0;
            while *path_ptr.add(len) != 0 {
                len += 1;
            }
    
            // Create a byte slice from the pointer
            let byte_slice = std::slice::from_raw_parts(path_ptr, len);
    
            // Convert the byte slice to a Rust string slice
            path_str = std::str::from_utf8(byte_slice).unwrap();

            let mut i = 0;

            // Iterate over argv until we encounter a NULL pointer
            loop {
                let c_str = *(argv_ptr as *const i32).add(i) as *const i32;

                if c_str.is_null() {
                    break;  // Stop if we encounter NULL
                }

                let arg_ptr = ((address as i64) + (c_str as i64)) as *const c_char;

                // Convert it to a Rust String
                let arg = CStr::from_ptr(arg_ptr)
                    .to_string_lossy()
                    .into_owned();
                args.push(arg);

                i += 1;  // Move to the next argument
            }
    
            // println!("path: {}", path_str);
            // println!("args: {:?}", args);
        }

        // get the stack pointer global
        let stack_pointer;
        if let Some(sp_extern) = caller.get_export("__stack_pointer") {
            match sp_extern {
                Extern::Global(sp) => {
                    match sp.get(&mut caller) {
                        Val::I32(val) => {
                            stack_pointer = val;
                        }
                        _ => {
                            println!("__stack_pointer export is not an i32");
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("__stack_pointer export is not a Global");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("__stack_pointer export not found");
            return Ok(-1);
        }

        // start unwind
        if let Some(asyncify_start_unwind_extern) = caller.get_export(ASYNCIFY_START_UNWIND) {
            match asyncify_start_unwind_extern {
                Extern::Func(asyncify_start_unwind) => {
                    match asyncify_start_unwind.typed::<i32, ()>(&caller) {
                        Ok(func) => {
                            let unwind_pointer: u64 = 0;
                            // 8 because we need to store unwind_data_start and unwind_data_end
                            // at the beginning of the unwind stack as the parameter for asyncify_start_unwind
                            // each of them are u64, so together is 8 bytes
                            let unwind_data_start: u64 = unwind_pointer + 8;
                            let unwind_data_end: u64 = stack_pointer as u64;
    
                            unsafe {
                                *(address as *mut u64) = unwind_data_start;
                                *(address as *mut u64).add(1) = unwind_data_end;
                            }
    
                            let _res = func.call(&mut caller, unwind_pointer as i32);
                        }
                        Err(err) => {
                            println!("the signature of asyncify_start_unwind function is not correct: {:?}", err);
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("asyncify_start_unwind export is not a function");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("asyncify_start_unwind export not found");
            return Ok(-1);
        }

        // get the asyncify_stop_unwind and asyncify_start_rewind, which will later
        // be used when the unwind process finished
        let asyncify_stop_unwind_func;

        if let Some(asyncify_stop_unwind_extern) = caller.get_export(ASYNCIFY_STOP_UNWIND) {
            match asyncify_stop_unwind_extern {
                Extern::Func(asyncify_stop_unwind) => {
                    match asyncify_stop_unwind.typed::<(), ()>(&caller) {
                        Ok(func) => {
                            asyncify_stop_unwind_func = func;
                        }
                        Err(err) => {
                            println!("the signature of asyncify_stop_unwind function is not correct: {:?}", err);
                            return Ok(-1);
                        }
                    }
                },
                _ => {
                    println!("asyncify_stop_unwind export is not a function");
                    return Ok(-1);
                }
            }
        }
        else {
            println!("asyncify_stop_unwind export not found");
            return Ok(-1);
        }

        let store = caller.as_context_mut().0;

        let cloned_run_command = self.run_command.clone();
        let cloned_next_cageid = self.next_cageid.clone();
        let cloned_lind_manager = self.lind_manager.clone();
        let cloned_pid = self.pid;

        store.set_on_called(Box::new(move |mut store| {
            // unwind finished and we need to stop the unwind
            let _res = asyncify_stop_unwind_func.call(&mut store, ());

            // lind_exec(cloned_pid as u64, cloned_pid as u64);
            let ret = exec(&cloned_run_command, path_str, &args, cloned_pid, &cloned_next_cageid, &cloned_lind_manager);

            return Ok(OnCalledAction::Finish(ret.expect("exec-ed module error")));
        }));

        return Ok(0);
    }

    pub fn getpid(&self) -> i32 {
        self.pid
    }

    fn next_cage_id(&self) -> Option<u64> {
        match self
            .next_cageid
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
                ..=0x1ffffffe => Some(v + 1),
                _ => None,
            }) {
            Ok(v) => Some(v + 1),
            Err(_) => None,
        }
    }

    fn fork_memory(&mut self, store: &StoreOpaque) {
        self.linker.allow_shadowing(true);
        for import in self.module.imports() {
            if let Some(m) = import.ty().memory() {
                if m.is_shared() {
                    let mem = SharedMemory::new(self.module.engine(), m.clone()).unwrap();
                    self.linker.define_with_inner(store, import.module(), import.name(), mem.clone()).unwrap();
                }
            }
        }
        self.linker.allow_shadowing(false);
    }

    pub fn fork(&self) -> Self {
        // let pid = match self.next_process_id() {
        //     Some(id) => id,
        //     None => {
        //         // to-do: should have a better error handling
        //         // instead of just panicking here
        //         panic!("running out of process id");
        //     }
        // };

        let forked_ctx = Self {
            // instance_pre: self.instance_pre.clone(),
            linker: self.linker.clone(),
            module: self.module.clone(),
            pid: 0,
            // next_pid: self.next_pid.clone(),
            next_cageid: self.next_cageid.clone(),
            lind_manager: self.lind_manager.clone(),
            run_command: self.run_command.clone()
        };

        return forked_ctx;
    }
}

pub fn add_to_linker<T: Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>(
    linker: &mut wasmtime::Linker<T>,
    get_cx: impl Fn(&T) -> &WasiForkCtx<T, U> + Send + Sync + Copy + 'static,
    get_cx_mut: impl Fn(&mut T) -> &mut WasiForkCtx<T, U> + Send + Sync + Copy + 'static,
    get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
    fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static,
    exec: impl Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>) -> Result<Vec<Val>> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    linker.func_wrap(
        "wasix",
        "lind-fork",
        move |mut caller: Caller<'_, T>, _: i32| -> i32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            match ctx.fork_call(&mut caller, get_cx_mut, get_wasi_cx, fork_host) {
                Ok(pid) => {
                    pid
                }
                Err(e) => {
                    log::error!("failed to fork: {}", e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "wasix",
        "lind-execv",
        move |mut caller: Caller<'_, T>, path: i64, argv: i64| -> i32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            match ctx.execv_call(&mut caller, path, argv, exec)  {
                Ok(ret) => {
                    ret
                }
                Err(e) => {
                    log::error!("failed to exec: {}", e);
                    -1
                }
            }
        },
    )?;

    Ok(())
}

fn support_asyncify(module: &Module) -> bool {
    module.get_export(ASYNCIFY_START_UNWIND).is_some() &&
    module.get_export(ASYNCIFY_STOP_UNWIND).is_some() &&
    module.get_export(ASYNCIFY_START_REWIND).is_some() &&
    module.get_export(ASYNCIFY_STOP_REWIND).is_some()
}

fn has_correct_signature(module: &Module) -> bool {
    if !match module.get_export(ASYNCIFY_START_UNWIND) {
        Some(ExternType::Func(ty)) => {
            ty.params().len() == 1
                && ty.params().nth(0).unwrap().is_i32()
                && ty.results().len() == 0
        }
        _ => false,
    } {
        return false;
    }
    if !match module.get_export(ASYNCIFY_STOP_UNWIND) {
        Some(ExternType::Func(ty)) => {
            ty.params().len() == 0
                && ty.results().len() == 0
        }
        _ => false,
    } {
        return false;
    }
    if !match module.get_export(ASYNCIFY_START_REWIND) {
        Some(ExternType::Func(ty)) => {
            ty.params().len() == 1
                && ty.params().nth(0).unwrap().is_i32()
                && ty.results().len() == 0
        }
        _ => false,
    } {
        return false;
    }
    if !match module.get_export(ASYNCIFY_STOP_REWIND) {
        Some(ExternType::Func(ty)) => {
            ty.params().len() == 0
                && ty.results().len() == 0
        }
        _ => false,
    } {
        return false;
    }

    true
}
