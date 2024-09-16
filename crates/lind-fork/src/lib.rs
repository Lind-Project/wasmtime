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
    // linker used by the module
    linker: Linker<T>,
    // the module associated with the ctx
    module: Module,

    // process id, should be same as cage id
    pid: i32,
    
    // next cage id
    next_cageid: Arc<AtomicU64>,

    // used to keep track of how many active cages are running
    lind_manager: Arc<LindCageManager>,

    // from run.rs, used for exec call
    run_command: U
}

impl<T: Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync> WasiForkCtx<T, U> {
    pub fn new(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U) -> Result<Self> {
        // this method should only be called once from run.rs, other instances of WasiForkCtx
        // are supposed to be created from fork() method
        
        // cage id starts from 1
        let pid = 1;
        let next_cageid = Arc::new(AtomicU64::new(1)); // cageid starts from 1
        Ok(Self { linker, module: module.clone(), pid, next_cageid, lind_manager: lind_manager.clone(), run_command })
    }

    // used by exec call
    pub fn new_with_pid(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U, pid: i32, next_cageid: Arc<AtomicU64>) -> Result<Self> {
        Ok(Self { linker, module: module.clone(), pid, next_cageid, lind_manager: lind_manager.clone(), run_command })
    }

    pub fn lind_syscall(&self, call_number: u32, call_name: u64, start_address: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64, arg6: u64) -> u32 {
        rustposix::lind_syscall_inner(self.pid as u64, call_number, call_name, start_address, arg1, arg2, arg3, arg4, arg5, arg6)
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
            // stop the rewind
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

            // retrieve the fork return value
            let retval = caller.as_context().get_rewinding_state().retval;

            // set rewinding state to false
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
        let parent_addr_len = defined_memory.current_length();

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
    
                            // store the parameter at the top of the stack
                            unsafe {
                                *(address as *mut u64) = unwind_data_start;
                                *(address as *mut u64).add(1) = unwind_data_end;
                            }
                            
                            // mark the start of unwind
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

        // we want to send this address to child thread
        let cloned_address = address as u64;

        // retrieve the child host
        let mut child_host = fork_host(caller.data());
        // get next cage id
        let child_cageid = self.next_cage_id();
        if let None = child_cageid {
            println!("running out of cageid!");
        }
        let child_cageid = child_cageid.unwrap();

        // set up child wasi_ctx cage id
        let child_wasi_ctx = get_wasi_cx(&mut child_host);
        let parent_cageid = child_wasi_ctx.get_lind_cageid();
        child_wasi_ctx.set_lind_cageid(child_cageid);

        // use the same engine for parent and child
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
                // create a new instance
                let store_inner = Store::<T>::new_inner(&engine);

                // get child context
                let child_ctx = get_cx(&mut child_host);
                child_ctx.pid = child_cageid as i32;

                // create a new memory area for child
                child_ctx.fork_memory(&store_inner, parent_addr_len);
                let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());

                let lind_manager = child_ctx.lind_manager.clone();
                let mut store = Store::new_with_inner(&engine, child_host, store_inner);

                // if parent is a thread, so does the child
                if is_parent_thread {
                    store.set_is_thread(true);
                }

                // instantiate the module
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
                    address_length = defined_memory.current_length();
                }

                // copy the entire memory area from parent to child
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

                // mark the child to rewind state
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
                    // main thread calls fork, then we calls from _start function
                    let child_start_func = instance
                        .get_func(&mut store, "_start")
                        .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();

                    let ty = child_start_func.ty(&store);

                    let values = Vec::new();
                    let mut results = vec![Val::null_func_ref(); ty.results().len()];

                    let _invoke_res = child_start_func
                        .call(&mut store, &values, &mut results);

                    // get the exit code of the module
                    let exit_code = results.get(0).expect("_start function does not have a return value");
                    match exit_code {
                        Val::I32(val) => {
                            // exit the cage with the exit code
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

            // mark the parent to rewind state
            let _ = asyncify_start_rewind_func.call(&mut store, rewind_pointer as i32);

            // set up rewind state and fork return value for parent
            store.set_rewinding_state(RewindingReturn {
                rewinding: true,
                retval: child_cageid as i32,
            });

            // return InvokeAgain here would make parent re-invoke main
            return Ok(OnCalledAction::InvokeAgain);
        }));

        // after returning from here, unwind process should start
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

        // parse the path and argv
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
    
                            // mark the state to unwind
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

            // to-do: exec should not change the process id/cage id, however, the exec call from rustposix takes an
            // argument to change the process id. If we pass the same cageid, it would cause some error
            // lind_exec(cloned_pid as u64, cloned_pid as u64);
            let ret = exec(&cloned_run_command, path_str, &args, cloned_pid, &cloned_next_cageid, &cloned_lind_manager);

            return Ok(OnCalledAction::Finish(ret.expect("exec-ed module error")));
        }));

        // after returning from here, unwind process should start
        return Ok(0);
    }

    pub fn exit_call(&self, mut caller: &mut Caller<'_, T>, code: i32) {
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
                            return;
                        }
                    }
                },
                _ => {
                    println!("__stack_pointer export is not a Global");
                    return;
                }
            }
        }
        else {
            println!("__stack_pointer export not found");
            return;
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
    
                            // mark the state to unwind
                            let _res = func.call(&mut caller, unwind_pointer as i32);
                        }
                        Err(err) => {
                            println!("the signature of asyncify_start_unwind function is not correct: {:?}", err);
                            return;
                        }
                    }
                },
                _ => {
                    println!("asyncify_start_unwind export is not a function");
                    return;
                }
            }
        }
        else {
            println!("asyncify_start_unwind export not found");
            return;
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
                            return;
                        }
                    }
                },
                _ => {
                    println!("asyncify_stop_unwind export is not a function");
                    return;
                }
            }
        }
        else {
            println!("asyncify_stop_unwind export not found");
            return;
        }

        let store = caller.as_context_mut().0;

        store.set_on_called(Box::new(move |mut store| {
            // unwind finished and we need to stop the unwind
            let _res = asyncify_stop_unwind_func.call(&mut store, ());

            // after unwind, just continue returning

            return Ok(OnCalledAction::Finish(vec![Val::I32(code)]));
        }));

        // after returning from here, unwind process should start
    }

    pub fn clone_call(&self, mut caller: &mut Caller<'_, T>, args: i32,
        get_cx: impl Fn(&mut T) -> &mut WasiForkCtx<T, U> + Send + Sync + Copy + 'static,
        get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
        fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static) -> Result<i32>
    {
        // get the base address of the memory
        let handle = caller.as_context().0.instance(InstanceId::from_index(0));
        let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
        let address = defined_memory.base;
        let parent_address_length = defined_memory.current_length();

        // we want to send this address to child thread
        let cloned_address = address as u64;

        // retrieve the child host
        let mut child_host = fork_host(caller.data());
        // get next cage id
        let child_cageid = self.next_cage_id();
        if let None = child_cageid {
            println!("running out of cageid!");
        }
        let child_cageid = child_cageid.unwrap();

        // set up child wasi_ctx cage id
        let child_wasi_ctx = get_wasi_cx(&mut child_host);
        let parent_cageid = child_wasi_ctx.get_lind_cageid();
        child_wasi_ctx.set_lind_cageid(child_cageid);

        // use the same engine for parent and child
        let engine = self.module.engine().clone();

        // set up unwind callback function
        let store = caller.as_context_mut().0;
        let is_parent_thread = store.is_thread();

        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);

        let builder = thread::Builder::new().name(format!("lind-fork-{}", child_cageid));
        builder.spawn(move || {
            // create a new instance
            let store_inner = Store::<T>::new_inner(&engine);

            // get child context
            let child_ctx = get_cx(&mut child_host);
            child_ctx.pid = child_cageid as i32;

            // create a new memory area for child
            child_ctx.fork_memory(&store_inner, parent_address_length);
            let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());

            let lind_manager = child_ctx.lind_manager.clone();
            let mut store = Store::new_with_inner(&engine, child_host, store_inner);

            // if parent is a thread, so does the child
            if is_parent_thread {
                store.set_is_thread(true);
            }

            // instantiate the module
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

            // copy the entire memory area from parent to child
            unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, child_address, address_length); }

            // new cage created, increment the cage counter
            lind_manager.increment();
            // create the cage in rustposix via rustposix fork
            lind_fork(parent_cageid, child_cageid);

            barrier_clone.wait();

            if store.is_thread() {
                // fork inside a thread is possible but not common
                // when fork happened inside a thread, it will only fork that specific thread
                // and left other threads un-copied.
                // to support this, we can just store the thread start args and calling wasi_thread_start
                // with the same start args here instead of _start entry.
                // however, since this is not a common practice, so we do not support this right now
                return -1;
            } else {
                let entry_point = instance
                    .get_typed_func::<i32, i32>(&mut store, "wasi_clone_start")
                    .unwrap();

                match entry_point.call(&mut store, args) {
                    Ok(exit_code) => { 
                        lind_exit(child_cageid, exit_code);
                    }
                    Err(e) => {
                        let e = wasi_common::maybe_exit_on_error(e);
                        eprintln!("Error: {:?}", e);
                        std::process::exit(1);
                    }
                }

                // the cage just exited, decrement the cage counter
                lind_manager.decrement();
            }

            return 0;
        }).unwrap();

        barrier.wait();

        // after returning from here, unwind process should start
        return Ok(child_cageid as i32);
    }

    pub fn clone2_call(&self, mut caller: &mut Caller<'_, T>,
                share_memory: bool,
                start_func: bool,
                args: Option<i32>,
                get_cx: impl Fn(&mut T) -> &mut WasiForkCtx<T, U> + Send + Sync + Copy + 'static,
                get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
                fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static) -> Result<i32> {
        // if share memory (i.e. thread), then start_func must be specified
        if share_memory && !start_func {
            return Ok(-1);
        }

        // if start_func exists, so does the args
        if start_func && args.is_none() {
            return Ok(-1);
        }
        
        // if there is no start_func, that would mean we are doing fork that requires to resume to
        // last execution position
        if !start_func {
            // if fork is called during the rewinding process
            // that would mean fork has completed and we want to stop the rewind
            // and return the fork result
            if caller.as_context().get_rewinding_state().rewinding {
                // stop the rewind
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

                // retrieve the fork return value
                let retval = caller.as_context().get_rewinding_state().retval;

                // set rewinding state to false
                caller.as_context_mut().set_rewinding_state(RewindingReturn {
                    rewinding: false,
                    retval: 0,
                });

                return Ok(retval);
            }
        }

        // get the base address of the memory
        let handle = caller.as_context().0.instance(InstanceId::from_index(0));
        let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
        let address = defined_memory.base;
        let parent_addr_len = defined_memory.current_length();

        // if no start_func, execution state must be resumed
        // we must do the unwind and rewind
        if !start_func {
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
        
                                // store the parameter at the top of the stack
                                unsafe {
                                    *(address as *mut u64) = unwind_data_start;
                                    *(address as *mut u64).add(1) = unwind_data_end;
                                }
                                
                                // mark the start of unwind
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

            // we want to send this address to child thread
            let cloned_address = address as u64;

            // retrieve the child host
            let mut child_host = fork_host(caller.data());
            // get next cage id
            let child_cageid = self.next_cage_id();
            if let None = child_cageid {
                println!("running out of cageid!");
            }
            let child_cageid = child_cageid.unwrap();

            // set up child wasi_ctx cage id
            let child_wasi_ctx = get_wasi_cx(&mut child_host);
            let parent_cageid = child_wasi_ctx.get_lind_cageid();
            child_wasi_ctx.set_lind_cageid(child_cageid);

            // use the same engine for parent and child
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
                    // create a new instance
                    let store_inner = Store::<T>::new_inner(&engine);

                    // get child context
                    let child_ctx = get_cx(&mut child_host);
                    child_ctx.pid = child_cageid as i32;

                    // create a new memory area for child
                    child_ctx.fork_memory(&store_inner, parent_addr_len);
                    let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());

                    let lind_manager = child_ctx.lind_manager.clone();
                    let mut store = Store::new_with_inner(&engine, child_host, store_inner);

                    // if parent is a thread, so does the child
                    if is_parent_thread {
                        store.set_is_thread(true);
                    }

                    // instantiate the module
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
                        address_length = defined_memory.current_length();
                    }

                    // copy the entire memory area from parent to child
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

                    // mark the child to rewind state
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
                        // main thread calls fork, then we calls from _start function
                        let child_start_func = instance
                            .get_func(&mut store, "_start")
                            .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();

                        let ty = child_start_func.ty(&store);

                        let values = Vec::new();
                        let mut results = vec![Val::null_func_ref(); ty.results().len()];

                        let _invoke_res = child_start_func
                            .call(&mut store, &values, &mut results);

                        // get the exit code of the module
                        let exit_code = results.get(0).expect("_start function does not have a return value");
                        match exit_code {
                            Val::I32(val) => {
                                // exit the cage with the exit code
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

                // mark the parent to rewind state
                let _ = asyncify_start_rewind_func.call(&mut store, rewind_pointer as i32);

                // set up rewind state and fork return value for parent
                store.set_rewinding_state(RewindingReturn {
                    rewinding: true,
                    retval: child_cageid as i32,
                });

                // return InvokeAgain here would make parent re-invoke main
                return Ok(OnCalledAction::InvokeAgain);
            }));

            // after returning from here, unwind process should start
            return Ok(0);
        }
        else {
            // otherwise, the spawned child does not need to resume execution state
            // it will just start from the specified start_func

            // we want to send this address to child thread
            let cloned_address = address as u64;

            // retrieve the child host
            let mut child_host = fork_host(caller.data());
            // get next cage id
            let child_cageid = self.next_cage_id();
            if let None = child_cageid {
                println!("running out of cageid!");
            }
            let child_cageid = child_cageid.unwrap();

            // set up child wasi_ctx cage id
            let child_wasi_ctx = get_wasi_cx(&mut child_host);
            let parent_cageid = child_wasi_ctx.get_lind_cageid();
            child_wasi_ctx.set_lind_cageid(child_cageid);

            // use the same engine for parent and child
            let engine = self.module.engine().clone();

            // set up unwind callback function
            let store = caller.as_context_mut().0;
            let is_parent_thread = store.is_thread();

            let barrier = Arc::new(Barrier::new(2));
            let barrier_clone = Arc::clone(&barrier);

            let builder = thread::Builder::new().name(format!("lind-fork-{}", child_cageid));
            builder.spawn(move || {
                // create a new instance
                let store_inner = Store::<T>::new_inner(&engine);

                // get child context
                let child_ctx = get_cx(&mut child_host);
                child_ctx.pid = child_cageid as i32;

                // create a new memory area for child
                child_ctx.fork_memory(&store_inner, parent_addr_len);
                let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());

                let lind_manager = child_ctx.lind_manager.clone();
                let mut store = Store::new_with_inner(&engine, child_host, store_inner);

                // if parent is a thread, so does the child
                if is_parent_thread {
                    store.set_is_thread(true);
                }

                // instantiate the module
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

                // copy the entire memory area from parent to child
                unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, child_address, address_length); }

                // new cage created, increment the cage counter
                lind_manager.increment();
                // create the cage in rustposix via rustposix fork
                lind_fork(parent_cageid, child_cageid);

                barrier_clone.wait();

                if store.is_thread() {
                    // fork inside a thread is possible but not common
                    // when fork happened inside a thread, it will only fork that specific thread
                    // and left other threads un-copied.
                    // to support this, we can just store the thread start args and calling wasi_thread_start
                    // with the same start args here instead of _start entry.
                    // however, since this is not a common practice, so we do not support this right now
                    return -1;
                } else {
                    let entry_point = instance
                        .get_typed_func::<i32, i32>(&mut store, "wasi_clone_start")
                        .unwrap();

                    match entry_point.call(&mut store, args.unwrap()) {
                        Ok(exit_code) => { 
                            lind_exit(child_cageid, exit_code);
                        }
                        Err(e) => {
                            let e = wasi_common::maybe_exit_on_error(e);
                            eprintln!("Error: {:?}", e);
                            std::process::exit(1);
                        }
                    }

                    // the cage just exited, decrement the cage counter
                    lind_manager.decrement();
                }

                return 0;
            }).unwrap();

            barrier.wait();

            return Ok(child_cageid as i32);
        }
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

    fn fork_memory(&mut self, store: &StoreOpaque, size: usize) {
        // allow shadowing means define a symbol that already exits would replace the old one
        self.linker.allow_shadowing(true);
        for import in self.module.imports() {
            if let Some(m) = import.ty().memory() {
                if m.is_shared() {
                    // define a new shared memory for the child
                    let mut plan = m.clone();
                    plan.set_minimum((size as u64).div_ceil(m.page_size()));

                    let mem = SharedMemory::new(self.module.engine(), plan.clone()).unwrap();
                    self.linker.define_with_inner(store, import.module(), import.name(), mem.clone()).unwrap();
                }
            }
        }
        // set shadowing state back
        self.linker.allow_shadowing(false);
    }

    pub fn fork(&self) -> Self {
        let forked_ctx = Self {
            linker: self.linker.clone(),
            module: self.module.clone(),
            pid: 0,
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
        "lind",
        "lind-syscall",
        move |mut caller: Caller<'_, T>, call_number: u32, call_name: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64, arg6: u64| -> u32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            let handle = caller.as_context().0.instance(InstanceId::from_index(0));
            let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
            let start_address = defined_memory.base as u64;

            ctx.lind_syscall(call_number, call_name, start_address, arg1, arg2, arg3, arg4, arg5, arg6)
        },
    )?;

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

    linker.func_wrap(
        "wasix",
        "lind-exit",
        move |mut caller: Caller<'_, T>, code: i32| {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            ctx.exit_call(&mut caller, code);
        },
    )?;

    linker.func_wrap(
        "wasix",
        "lind-clone",
        move |mut caller: Caller<'_, T>, args: i32| -> i32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            match ctx.clone_call(&mut caller, args, get_cx_mut, get_wasi_cx, fork_host) {
                Ok(pid) => {
                    pid
                }
                Err(e) => {
                    log::error!("failed to clone: {}", e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "wasix",
        "lind-do-clone",
        move |mut caller: Caller<'_, T>, args: i32, share_memory: i32, start_func: i32| -> i32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            let args = {
                if args == 0 { None }
                else { Some(args) }
            };

            let share_memory = {
                if share_memory == 0 { false }
                else { true }
            };

            let start_func = {
                if start_func == 0 { false }
                else { true }
            };

            println!("args: {:?}, share_memory: {}, start_func: {}", args, share_memory, start_func);

            match ctx.clone2_call(&mut caller, share_memory, start_func, args, get_cx_mut, get_wasi_cx, fork_host) {
                Ok(pid) => {
                    pid
                }
                Err(e) => {
                    log::error!("failed to clone: {}", e);
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
