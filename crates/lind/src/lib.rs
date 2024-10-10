#![allow(dead_code)]

use anyhow::{anyhow, Result};
use wasi_common::WasiCtx;
use wasmtime_lind_utils::{parse_env_var, LindCageManager};

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use wasmtime::{AsContext, AsContextMut, Caller, ExternType, Linker, Module, SharedMemory, Store, Val, Extern, OnCalledAction, RewindingReturn, StoreOpaque, InstanceId};

use wasmtime_environ::MemoryIndex;

const ASYNCIFY_START_UNWIND: &str = "asyncify_start_unwind";
const ASYNCIFY_STOP_UNWIND: &str = "asyncify_stop_unwind";
const ASYNCIFY_START_REWIND: &str = "asyncify_start_rewind";
const ASYNCIFY_STOP_REWIND: &str = "asyncify_stop_rewind";

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct CloneArgStruct {
    pub flags: u64,           // Flags that control the behavior of the child process
    pub pidfd: u64,           // File descriptor to receive the child's PID
    pub child_tid: u64,       // Pointer to a memory location where the child TID will be stored
    pub parent_tid: u64,      // Pointer to a memory location where the parent's TID will be stored
    pub exit_signal: u64,     // Signal to be sent when the child process exits
    pub stack: u64,           // Address of the stack for the child process
    pub stack_size: u64,      // Size of the stack for the child process
    pub tls: u64,             // Thread-Local Storage (TLS) descriptor for the child thread
    pub set_tid: u64,         // Pointer to an array of TIDs to be set in the child
    pub set_tid_size: u64,    // Number of TIDs in the `set_tid` array
    pub cgroup: u64,          // File descriptor for the cgroup to which the child process should be attached
}

/* Cloning flags.  */
pub const CSIGNAL: u64 =       0x000000ff; /* Signal mask to be sent at exit.  */
pub const CLONE_VM: u64 =      0x00000100; /* Set if VM shared between processes.  */
pub const CLONE_FS: u64 =      0x00000200; /* Set if fs info shared between processes.  */
pub const CLONE_FILES: u64 =   0x00000400; /* Set if open files shared between processes.  */
pub const CLONE_SIGHAND: u64 = 0x00000800; /* Set if signal handlers shared.  */
pub const CLONE_PIDFD: u64 =   0x00001000; /* Set if a pidfd should be placed in parent.  */
pub const CLONE_PTRACE: u64 =  0x00002000; /* Set if tracing continues on the child.  */
pub const CLONE_VFORK: u64 =   0x00004000; /* Set if the parent wants the child to wake it up on mm_release.  */
pub const CLONE_PARENT: u64 =  0x00008000; /* Set if we want to have the same parent as the cloner.  */
pub const CLONE_THREAD: u64 =  0x00010000; /* Set to add to same thread group.  */
pub const CLONE_NEWNS: u64 =   0x00020000; /* Set to create new namespace.  */
pub const CLONE_SYSVSEM: u64 = 0x00040000; /* Set to shared SVID SEM_UNDO semantics.  */
pub const CLONE_SETTLS: u64 =  0x00080000; /* Set TLS info.  */
pub const CLONE_PARENT_SETTID: u64 = 0x00100000; /* Store TID in userlevel buffer before MM copy.  */
pub const CLONE_CHILD_CLEARTID: u64 = 0x00200000; /* Register exit futex and memory location to clear.  */
pub const CLONE_DETACHED: u64 = 0x00400000; /* Create clone detached.  */
pub const CLONE_UNTRACED: u64 = 0x00800000; /* Set if the tracing process can't force CLONE_PTRACE on this clone.  */
pub const CLONE_CHILD_SETTID: u64 = 0x01000000; /* Store TID in userlevel buffer in the child.  */
pub const CLONE_NEWCGROUP: u64 =    0x02000000;	/* New cgroup namespace.  */
pub const CLONE_NEWUTS: u64 =	0x04000000;	/* New utsname group.  */
pub const CLONE_NEWIPC: u64 =	0x08000000;	/* New ipcs.  */
pub const CLONE_NEWUSER: u64 =	0x10000000;	/* New user namespace.  */
pub const CLONE_NEWPID: u64 =	0x20000000;	/* New pid namespace.  */
pub const CLONE_NEWNET: u64 =	0x40000000;	/* New network namespace.  */
pub const CLONE_IO: u64 =	0x80000000;	/* Clone I/O context.  */
/* cloning flags intersect with CSIGNAL so can be used only with unshare and
   clone3 syscalls.  */
pub const CLONE_NEWTIME: u64 =	0x00000080;      /* New time namespace */

fn print_memory_segment(addr: *const u8, length: usize, bytes_per_line: usize) {
    unsafe {
        let slice = std::slice::from_raw_parts(addr, length);
        for (i, byte) in slice.iter().enumerate() {
            print!("{:02X} ", byte);
            if (i + 1) % bytes_per_line == 0 {
                println!(); // Newline after every `bytes_per_line` bytes
            }
        }
        if length % bytes_per_line != 0 {
            println!(); // Final newline if the last line is incomplete
        }
    }
}

// Define the trait with the required method
pub trait LindHost<T, U> {
    fn get_ctx(&self) -> LindCtx<T, U>;
}

#[derive(Clone)]
pub struct LindCtx<T, U> {
    // linker used by the module
    linker: Linker<T>,
    // the module associated with the ctx
    module: Module,

    // process id, should be same as cage id
    pid: i32,
    
    // next cage id
    next_cageid: Arc<AtomicU64>,

    // next thread id
    next_threadid: Arc<AtomicU32>,

    // used to keep track of how many active cages are running
    lind_manager: Arc<LindCageManager>,

    // from run.rs, used for exec call
    run_command: U,

    // get LindCtx from host
    get_cx: Arc<dyn Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + 'static>,

    // fork the host
    fork_host: Arc<dyn Fn(&T) -> T + Send + Sync + 'static>,

    // exec the host
    exec_host: Arc<dyn Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>, &Option<Vec<(String, Option<String>)>>) -> Result<Vec<Val>> + Send + Sync + 'static>,
}

impl<T: Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync> LindCtx<T, U> {
    pub fn new(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U,
               next_cageid: Arc<AtomicU64>,
               get_cx: impl Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + 'static,
               fork_host: impl Fn(&T) -> T + Send + Sync + 'static,
               exec: impl Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>, &Option<Vec<(String, Option<String>)>>) -> Result<Vec<Val>> + Send + Sync + 'static,
            ) -> Result<Self> {
        // this method should only be called once from run.rs, other instances of LindCtx
        // are supposed to be created from fork() method

        let get_cx = Arc::new(get_cx);
        let fork_host = Arc::new(fork_host);
        let exec_host = Arc::new(exec);
        
        // cage id starts from 1
        let pid = 1;
        // let next_cageid = Arc::new(AtomicU64::new(1)); // cageid starts from 1
        let next_threadid = Arc::new(AtomicU32::new(1)); // cageid starts from 1
        Ok(Self { linker, module: module.clone(), pid, next_cageid, next_threadid, lind_manager: lind_manager.clone(), run_command, get_cx, fork_host, exec_host })
    }

    // used by exec call
    pub fn new_with_pid(module: Module, linker: Linker<T>, lind_manager: Arc<LindCageManager>, run_command: U, pid: i32, next_cageid: Arc<AtomicU64>,
                        get_cx: impl Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + 'static,
                        fork_host: impl Fn(&T) -> T + Send + Sync + 'static,
                        exec: impl Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>, &Option<Vec<(String, Option<String>)>>) -> Result<Vec<Val>> + Send + Sync + 'static,
        ) -> Result<Self> {

        let get_cx = Arc::new(get_cx);
        let fork_host = Arc::new(fork_host);
        let exec_host = Arc::new(exec);

        let next_threadid = Arc::new(AtomicU32::new(1)); // cageid starts from 1

        Ok(Self { linker, module: module.clone(), pid, next_cageid, next_threadid, lind_manager: lind_manager.clone(), run_command, get_cx, fork_host, exec_host })
    }

    // pub fn lind_syscall(&self, call_number: u32, call_name: u64, start_address: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64, arg6: u64) -> i32 {
    //     // rustposix::lind_syscall_inner(self.pid as u64, call_number, call_name, start_address, arg1, arg2, arg3, arg4, arg5, arg6)
    //     rawposix::lind_syscall_inner(self.pid as u64, call_number, call_name, start_address, arg1, arg2, arg3, arg4, arg5, arg6)
    // }

    pub fn catch_rewind(&self, mut caller: &mut Caller<'_, T>) -> Result<i32> {
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

            // let set_stack_pointer_func;
            // if let Some(set_stack_pointer_extern) = caller.get_export("set_stack_pointer") {
            //     match set_stack_pointer_extern {
            //         Extern::Func(set_stack_pointer) => {
            //             match set_stack_pointer.typed::<i32, ()>(&caller) {
            //                 Ok(func) => {
            //                     set_stack_pointer_func = func;
            //                 }
            //                 Err(err) => {
            //                     println!("the signature of set_stack_pointer function is not correct: {:?}", err);
            //                     return Ok(-1);
            //                 }
            //             }
            //         },
            //         _ => {
            //             println!("set_stack_pointer export is not a function");
            //             return Ok(-1);
            //         }
            //     }
            // }
            // else {
            //     println!("set_stack_pointer export not found");
            //     return Ok(-1);
            // }

            // // get the stack pointer global
            // let new_stack_pointer;
            // if let Some(sp_extern) = caller.get_export("__stack_pointer") {
            //     match sp_extern {
            //         Extern::Global(sp) => {
            //             match sp.get(&mut caller) {
            //                 Val::I32(val) => {
            //                     new_stack_pointer = val;
            //                 }
            //                 _ => {
            //                     println!("__stack_pointer export is not an i32");
            //                     return Ok(-1);
            //                 }
            //             }
            //         },
            //         _ => {
            //             println!("__stack_pointer export is not a Global");
            //             return Ok(-1);
            //         }
            //     }
            // }
            // else {
            //     println!("__stack_pointer export not found");
            //     return Ok(-1);
            // }

            // if retval != 0 {
            //     println!("after rewind, parent stack pointer: {}", new_stack_pointer);
            // } else {
            //     println!("after rewind, child stack pointer: {}", new_stack_pointer);
            //     // let _ = set_stack_pointer_func.call(&mut caller, 200000);
            // }

            // if retval != 0 {
            //     loop {

            //     }
            // }

            return Ok(retval);
        }

        Ok(-1)
    }

    pub fn fork_call(&self, mut caller: &mut Caller<'_, T>
                ) -> Result<i32> {
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
        // if caller.as_context().get_rewinding_state().rewinding {
        //     // stop the rewind
        //     if let Some(asyncify_stop_rewind_extern) = caller.get_export(ASYNCIFY_STOP_REWIND) {
        //         match asyncify_stop_rewind_extern {
        //             Extern::Func(asyncify_stop_rewind) => {
        //                 match asyncify_stop_rewind.typed::<(), ()>(&caller) {
        //                     Ok(func) => {
        //                         let _res = func.call(&mut caller, ());
        //                     }
        //                     Err(err) => {
        //                         println!("the signature of asyncify_stop_rewind is not correct: {:?}", err);
        //                         return Ok(-1);
        //                     }
        //                 }
        //             },
        //             _ => {
        //                 println!("asyncify_stop_rewind export is not a function");
        //                 return Ok(-1);
        //             }
        //         }
        //     }
        //     else {
        //         println!("asyncify_stop_rewind export not found");
        //         return Ok(-1);
        //     }

        //     // retrieve the fork return value
        //     let retval = caller.as_context().get_rewinding_state().retval;

        //     // set rewinding state to false
        //     caller.as_context_mut().set_rewinding_state(RewindingReturn {
        //         rewinding: false,
        //         retval: 0,
        //     });

        //     return Ok(retval);
        // }

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
        let mut child_host = (self.fork_host)(caller.data());
        // get next cage id
        let child_cageid = self.next_cage_id();
        if let None = child_cageid {
            println!("running out of cageid!");
        }
        let child_cageid = child_cageid.unwrap();
        rawposix::lind_fork(self.pid as u64, child_cageid);

        // use the same engine for parent and child
        let engine = self.module.engine().clone();

        let get_cx = self.get_cx.clone();
        let parent_pid = self.pid;

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
                // lind_fork(parent_pid as u64, child_cageid);

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

                    let invoke_res = child_start_func
                        .call(&mut store, &values, &mut results);

                    if let Err(err) = invoke_res {
                        let e = wasi_common::maybe_exit_on_error(err);
                        eprintln!("Error: {:?}", e);
                        return 0;
                    }

                    // get the exit code of the module
                    let exit_code = results.get(0).expect("_start function does not have a return value");
                    match exit_code {
                        Val::I32(val) => {
                            // exit the cage with the exit code
                            rawposix::lind_exit(child_cageid, *val);
                            // let _ = on_child_exit(*val);
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

    pub fn fork_shared_call(&self, mut caller: &mut Caller<'_, T>,
                    stack_addr: i32, stack_size: i32, child_tid: u64
                ) -> Result<i32> {
        // if !support_asyncify(instance_pre.module()) {
        //     log::error!("failed to find asyncify functions");
        //     return Ok(-1);
        // }
        // if !has_correct_signature(instance_pre.module()) {
        //     log::error!("the exported asyncify functions have incorrect signatures");
        //     return Ok(-1);
        // }

        // get the base address of the memory
        let handle = caller.as_context().0.instance(InstanceId::from_index(0));
        let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
        let address = defined_memory.base;
        let parent_addr_len = defined_memory.current_length();

        let parent_stack_base = caller.as_context().get_stack_top();

        // get the stack pointer global
        let stack_pointer = caller.get_stack_pointer().unwrap();

        // start unwind
        if let Some(asyncify_start_unwind_extern) = caller.get_export(ASYNCIFY_START_UNWIND) {
            match asyncify_start_unwind_extern {
                Extern::Func(asyncify_start_unwind) => {
                    match asyncify_start_unwind.typed::<i32, ()>(&caller) {
                        Ok(func) => {
                            let unwind_pointer: u64 = parent_stack_base;
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
        let parent_stack_bottom = caller.as_context().get_stack_base();

        // retrieve the child host
        let mut child_host = caller.data().clone();
        // get next cage id
        let child_cageid = self.pid;

        // use the same engine for parent and child
        let engine = self.module.engine().clone();

        let get_cx = self.get_cx.clone();
        let parent_pid = self.pid;

        // set up child_tid
        let next_tid = match self.next_thread_id() {
            Some(val) => val,
            None => {
                println!("running out of thread id!");
                0
            }
        };
        let child_tid = child_tid as *mut u32;
        unsafe { *child_tid = next_tid; }

        // set up unwind callback function
        let store = caller.as_context_mut().0;
        let is_parent_thread = store.is_thread();
        store.set_on_called(Box::new(move |mut store| {
            let unwind_stack_finish;

            let address = cloned_address as *mut u64;
            let unwind_start_address = (cloned_address + 8) as *mut u64;

            unsafe {
                unwind_stack_finish = *address;
            }

            // let unwind_size = unwind_stack_finish - 8;
            // let mut unwind_stack = Vec::with_capacity(unwind_size as usize);

            // unsafe {
            //     let src_slice = std::slice::from_raw_parts(unwind_start_address as *mut u8, unwind_size as usize);
            //     unwind_stack.extend_from_slice(src_slice);
            // }

            // unwind finished and we need to stop the unwind
            let _res = asyncify_stop_unwind_func.call(&mut store, ());

            let rewind_base = parent_stack_base;

            let rewind_pointer: u64 = rewind_base;
            let rewind_pointer_child = stack_addr as u64 - stack_size as u64;

            let rewind_start_parent = (cloned_address + rewind_pointer) as *mut u8;
            let rewind_start_child = (cloned_address + rewind_pointer_child) as *mut u8;
            let rewind_total_size = (unwind_stack_finish - rewind_base) as usize;
            // copy the unwind data to child stack
            unsafe { std::ptr::copy_nonoverlapping(rewind_start_parent, rewind_start_child, rewind_total_size); }
            unsafe {
                // value used to restore the stack pointer is stored at offset of 0xc (12) from unwind data start
                // let's retrieve it
                let stack_pointer_address = rewind_start_child.add(12) as *mut u32;
                // offset = parent's stack bottom - stored sp (how far is stored sp from parent's stack bottom)
                println!("parent_stack_bottom: {}, stored sp: {}, stack_addr: {}", parent_stack_bottom, *stack_pointer_address, stack_addr);
                let offset = parent_stack_bottom as u32 - *stack_pointer_address;
                // child stored sp = child's stack bottom - offset = child's stack bottom - (parent's stack bottom - stored sp)
                // child stored sp = child's stack bottom - parent's stack bottom + stored sp
                // keep child's stored sp same distance from its stack bottom
                let child_sp_val = stack_addr as u32 - offset;
                // replace the stored stack pointer in child's unwind data
                *stack_pointer_address = child_sp_val;

                // first 4 bytes in unwind data represent the address of the end of the unwind data
                // we also need to change this for child
                let child_rewind_data_start = *(rewind_start_child as *mut u32) + rewind_pointer_child as u32;

                *(rewind_start_child as *mut u32) = child_rewind_data_start;
            }

            let builder = thread::Builder::new().name(format!("lind-thread-{}", next_tid));
            builder.spawn(move || {
                // create a new instance
                let store_inner = Store::<T>::new_inner(&engine);

                // get child context
                let child_ctx = get_cx(&mut child_host);
                child_ctx.pid = child_cageid as i32;

                // create a new memory area for child
                // child_ctx.fork_memory(&store_inner, parent_addr_len);
                let instance_pre = Arc::new(child_ctx.linker.instantiate_pre(&child_ctx.module).unwrap());

                let mut store = Store::new_with_inner(&engine, child_host, store_inner);

                // if parent is a thread, so does the child
                if is_parent_thread {
                    store.set_is_thread(true);
                }

                // instantiate the module
                let instance = instance_pre.instantiate(&mut store).unwrap();

                // we might also want to perserve the offset of current stack pointer to stack bottom
                // not very sure if this is required, but just keep everything the same from parent seems to be good
                let offset = parent_stack_base as i32 - stack_pointer;
                let stack_pointer_setter = instance
                    .get_typed_func::<(i32), ()>(&mut store, "set_stack_pointer")
                    .unwrap();
                let _ = stack_pointer_setter.call(&mut store, stack_addr - offset);

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
                let _ = child_rewind_start.call(&mut store, rewind_pointer_child as i32);

                // set up rewind state and fork return value for child
                store.as_context_mut().set_rewinding_state(RewindingReturn {
                    rewinding: true,
                    retval: 0,
                });

                // set stack base for child
                store.as_context_mut().set_stack_top(rewind_pointer_child);

                // main thread calls fork, then we calls from _start function
                let child_start_func = instance
                    .get_func(&mut store, "_start")
                    .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();

                let ty = child_start_func.ty(&store);

                let values = Vec::new();
                let mut results = vec![Val::null_func_ref(); ty.results().len()];

                let invoke_res = child_start_func
                    .call(&mut store, &values, &mut results);

                if let Err(err) = invoke_res {
                    let e = wasi_common::maybe_exit_on_error(err);
                    eprintln!("Error: {:?}", e);
                    return 0;
                }

                // get the exit code of the module
                let exit_code = results.get(0).expect("_start function does not have a return value");
                match exit_code {
                    Val::I32(val) => {
                        // technically we need to do some clean up here if necessary
                        // like clean up signal stuff
                        // but signal is still WIP so this is a placeholder for this in the future
                    },
                    _ => {
                        println!("unexpected _start function return type: {:?}", exit_code);
                    }
                }

                return 0;
            }).unwrap();

            // mark the parent to rewind state
            let _ = asyncify_start_rewind_func.call(&mut store, rewind_pointer as i32);

            // set up rewind state and fork return value for parent
            store.set_rewinding_state(RewindingReturn {
                rewinding: true,
                retval: next_tid as i32,
            });


            // loop {}

            // return InvokeAgain here would make parent re-invoke main
            return Ok(OnCalledAction::InvokeAgain);
        }));

        // after returning from here, unwind process should start
        return Ok(0);
    }

    pub fn execve_call(&self, mut caller: &mut Caller<'_, T>,
                             path: i64,
                             argv: i64,
                             envs: Option<i64>
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
        let mut environs = None;

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

        if let Some(envs_addr) = envs {
            let env_ptr = ((address as i64) + envs_addr) as *const *const u8;
            let mut env_vec = Vec::new();

            unsafe {
                let mut i = 0;
    
                // Iterate over argv until we encounter a NULL pointer
                loop {
                    let c_str = *(env_ptr as *const i32).add(i) as *const i32;
    
                    if c_str.is_null() {
                        break;  // Stop if we encounter NULL
                    }
    
                    let env_ptr = ((address as i64) + (c_str as i64)) as *const c_char;
    
                    // Convert it to a Rust String
                    let env = CStr::from_ptr(env_ptr)
                        .to_string_lossy()
                        .into_owned();
                    let parsed = parse_env_var(&env);
                    env_vec.push(parsed);
    
                    i += 1;  // Move to the next argument
                }
            }
            environs = Some(env_vec);
        }

        // println!("args: {:?}, envs: {:?}", args, environs);

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

        let exec_call = self.exec_host.clone();

        store.set_on_called(Box::new(move |mut store| {
            // unwind finished and we need to stop the unwind
            let _res = asyncify_stop_unwind_func.call(&mut store, ());

            // to-do: exec should not change the process id/cage id, however, the exec call from rustposix takes an
            // argument to change the process id. If we pass the same cageid, it would cause some error
            // lind_exec(cloned_pid as u64, cloned_pid as u64);
            let ret = exec_call(&cloned_run_command, path_str, &args, cloned_pid, &cloned_next_cageid, &cloned_lind_manager, &environs);

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

    pub fn clone_call(&self, caller: &mut Caller<'_, T>, args: i32,
        get_cx: impl Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + Copy + 'static,
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
            // lind_fork(parent_cageid, child_cageid);

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
                        // lind_exit(child_cageid, exit_code);
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
                get_cx: impl Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + Copy + 'static,
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
                    // lind_fork(parent_cageid, child_cageid);

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
                                // lind_exit(child_cageid, *val);
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
                // lind_fork(parent_cageid, child_cageid);

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
                            // lind_exit(child_cageid, exit_code);
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
        return Some(self.next_cageid.load(Ordering::SeqCst));
        // match self
        //     .next_cageid
        //     .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
        //         ..=0x1ffffffe => Some(v + 1),
        //         _ => None,
        //     }) {
        //     Ok(v) => Some(v + 1),
        //     Err(_) => None,
        // }
    }

    fn next_thread_id(&self) -> Option<u32> {
        match self
            .next_threadid
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
                ..=0x1ffffffe => Some(v + 1),
                _ => None,
            }) {
            Ok(v) => Some(v + 1),
            Err(_) => None,
        }
    }

    fn fork_memory(&mut self, store: &StoreOpaque, size: usize) {
        // allow shadowing means defining a symbol that already exits would replace the old one
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
            next_threadid: Arc::new(AtomicU32::new(1)),
            lind_manager: self.lind_manager.clone(),
            run_command: self.run_command.clone(),
            get_cx: self.get_cx.clone(),
            fork_host: self.fork_host.clone(),
            exec_host: self.exec_host.clone()
        };

        return forked_ctx;
    }
}

pub fn add_to_linker<T: Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>(
    linker: &mut wasmtime::Linker<T>,
    get_cx: impl Fn(&T) -> &LindCtx<T, U> + Send + Sync + Copy + 'static,
    get_cx_mut: impl Fn(&mut T) -> &mut LindCtx<T, U> + Send + Sync + Copy + 'static,
    get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
    fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static,
    exec: impl Fn(&U, &str, &Vec<String>, i32, &Arc<AtomicU64>, &Arc<LindCageManager>, &Option<Vec<(String, Option<String>)>>) -> Result<Vec<Val>> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    // linker.func_wrap(
    //     "lind",
    //     "lind-syscall",
    //     move |caller: Caller<'_, T>, call_number: u32, call_name: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64, arg6: u64| -> i32 {
    //         println!("lind syscall");
    //         let host = caller.data().clone();
    //         let ctx = get_cx(&host);

    //         let handle = caller.as_context().0.instance(InstanceId::from_index(0));
    //         let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
    //         let start_address = defined_memory.base as u64;

    //         ctx.lind_syscall(call_number, call_name, start_address, arg1, arg2, arg3, arg4, arg5, arg6)
    //     },
    // )?;

    // linker.func_wrap(
    //     "wasix",
    //     "lind-fork",
    //     move |mut caller: Caller<'_, T>, _: i32| -> i32 {
    //         let host = caller.data().clone();
    //         let ctx = get_cx(&host);

    //         match ctx.fork_call(&mut caller) {
    //             Ok(pid) => {
    //                 pid
    //             }
    //             Err(e) => {
    //                 log::error!("failed to fork: {}", e);
    //                 -1
    //             }
    //         }
    //     },
    // )?;

    // linker.func_wrap(
    //     "wasix",
    //     "lind-execv",
    //     move |mut caller: Caller<'_, T>, path: i64, argv: i64| -> i32 {
    //         let host = caller.data().clone();
    //         let ctx = get_cx(&host);

    //         match ctx.execve_call(&mut caller, path, argv, exec)  {
    //             Ok(ret) => {
    //                 ret
    //             }
    //             Err(e) => {
    //                 log::error!("failed to exec: {}", e);
    //                 -1
    //             }
    //         }
    //     },
    // )?;

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

pub fn get_memory_base<T: Clone + Send + 'static + std::marker::Sync>(caller: &Caller<'_, T>) -> u64 {
    let handle = caller.as_context().0.instance(InstanceId::from_index(0));
    let defined_memory = handle.get_memory(MemoryIndex::from_u32(0));
    defined_memory.base as u64
}

pub fn lind_fork<T: LindHost<T, U> + Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>
        (caller: &mut Caller<'_, T>) -> Result<i32> {
    let host = caller.data().clone();
    let ctx = host.get_ctx();
    ctx.fork_call(caller)
}

pub fn lind_fork_shared_memory<T: LindHost<T, U> + Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>
        (caller: &mut Caller<'_, T>,
        stack_addr: i32, stack_size: i32, child_tid: u64) -> Result<i32> {
    let host = caller.data().clone();
    let ctx = host.get_ctx();
    ctx.fork_shared_call(caller, stack_addr, stack_size, child_tid)
}

pub fn catch_rewind<T: LindHost<T, U> + Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>(caller: &mut Caller<'_, T>) -> Result<i32> {
    let host = caller.data().clone();
    let ctx = host.get_ctx();
    ctx.catch_rewind(caller)
}

pub fn clone_syscall<T: LindHost<T, U> + Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>
        (caller: &mut Caller<'_, T>, args: &mut CloneArgStruct) -> i32
{
    let rewind_res = match catch_rewind(caller) {
        Ok(val) => val,
        Err(_) => -1
    };

    if rewind_res >= 0 { return rewind_res; }

    // get the flags
    let flags = args.flags;
    // if CLONE_VM is set, we are creating a new thread (i.e. pthread_create)
    // otherwise, we are creating a process (i.e. fork)
    let isthread = flags & (CLONE_VM as u64);

    if isthread == 0 {
        match lind_fork(caller) {
            Ok(res) => res,
            Err(e) => -1
        }
    }
    else {
        // pthread_create
        match lind_fork_shared_memory(caller, args.stack as i32, args.stack_size as i32, args.child_tid) {
            Ok(res) => res,
            Err(e) => -1
        }
    }
}

pub fn exec_syscall<T: LindHost<T, U> + Clone + Send + 'static + std::marker::Sync, U: Clone + Send + 'static + std::marker::Sync>
        (caller: &mut Caller<'_, T>, path: i64, argv: i64, envs: i64) -> i32 {
    let host = caller.data().clone();
    let ctx = host.get_ctx();

    match ctx.execve_call(caller, path, argv, Some(envs))  {
        Ok(ret) => {
            ret
        }
        Err(e) => {
            log::error!("failed to exec: {}", e);
            -1
        }
    }
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
