use anyhow::{anyhow, Result};
use rustposix::lind_syscall_inner;
use wasi_common::WasiCtx;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Barrier};
use std::thread;
use wasmtime::{AsContext, AsContextMut, Caller, ExternType, Func, InstancePre, Linker, Module, SharedMemory, Store, StoreContextMut, TypedFunc, Val};

use wasmtime::runtime::Extern;
use wasmtime::runtime::OnCalledAction;
use wasmtime::runtime::RewindingReturn;

use wasmtime_environ::MemoryIndex;

const ASYNCIFY_START_UNWIND: &str = "asyncify_start_unwind";
const ASYNCIFY_STOP_UNWIND: &str = "asyncify_stop_unwind";
const ASYNCIFY_START_REWIND: &str = "asyncify_start_rewind";
const ASYNCIFY_STOP_REWIND: &str = "asyncify_stop_rewind";

pub struct WasiForkCtx<T> {
    // instance_pre seems to be a static data and won't be modified anymore once created
    // so we may not need to make a deep clone of it
    instance_pre: Arc<InstancePre<T>>,
    
    // cageid is stored at preview1_ctx, so we do not need a duplicated field here
    // cageid: u64,

    // current pid associated with this ctx
    pid: i32,
    // next_pid and next_cageid should be shared between processes
    next_pid: Arc<AtomicI32>,
    next_cageid: Arc<AtomicU64>,
}

impl<T: Clone + Send + 'static + std::marker::Sync> WasiForkCtx<T> {
    pub fn new(module: Module, linker: Arc<Linker<T>>) -> Result<Self> {
        // this method should only be called once from run.rs, other instances of WasiForkCtx
        // are supposed to be created from fork() method
        let instance_pre = Arc::new(linker.instantiate_pre(&module)?).clone();
        let pid = 0;
        let next_pid = Arc::new(AtomicI32::new(0));
        let next_cageid = Arc::new(AtomicU64::new(1)); // cageid starts from 1
        Ok(Self { instance_pre, pid, next_pid, next_cageid })
    }

    pub fn fork_call(&self, mut caller: &mut Caller<'_, T>,
                get_cx: impl Fn(&T) -> &WasiForkCtx<T> + Send + Sync + Copy + 'static,
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
        if caller.store.0.inner.rewinding.rewinding {
            if let Some(asyncify_stop_rewind_extern) = caller.get_export(ASYNCIFY_STOP_REWIND) {
                match asyncify_stop_rewind_extern {
                    Extern::Func(asyncify_stop_rewind) => {
                        match asyncify_stop_rewind.typed::<(), ()>(&caller) {
                            Ok(func) => {
                                func.call(&mut caller, ());
                            }
                            Err(err) => {
                                println!("something went wrong");
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

            let retval = caller.store.0.inner.rewinding.retval;

            caller.store.0.inner.rewinding = RewindingReturn {
                rewinding: false,
                retval: 0,
            };

            return Ok(retval);
        }

        // let mut child_host = caller.data().clone();
        let mut child_host = fork_host(caller.data());
        let mut child_ctx = get_cx(&child_host);
        let child_pid = child_ctx.pid;

        // get the base address of the memory
        let address;
        if let Some(instance_item) = caller.store.as_context().0.instances.get(0) {
            let vm_instance = instance_item.handle.instance();
            let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
            address = defined_memory.base;
        } else {
            println!("no memory found");
            return Ok(-1);
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
    
                            func.call(&mut caller, (unwind_pointer as i32));
                        }
                        Err(err) => {
                            println!("something went wrong");
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
                            println!("something went wrong");
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
                            println!("something went wrong");
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
        let instance_pre = self.instance_pre.clone();

        // set up child wasi_ctx cage id
        let child_wasi_ctx = get_wasi_cx(&mut child_host);
        let parent_cageid = child_wasi_ctx.get_lind_cageid();
        // let child_cageid = parent_cageid + 1;
        let child_cageid = self.next_cage_id();
        if let None = child_cageid {
            println!("running out of cageid!");
            return Ok(-1);
        }
        let child_cageid = child_cageid.unwrap();
        child_wasi_ctx.set_lind_cageid(child_cageid);
        
        // set up unwind callback function
        let store = &mut caller.store.0;
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
            asyncify_stop_unwind_func.call(&mut store, ());

            let rewind_pointer: u64 = 0;

            // use a barrier to make sure the child has fully copied parent's memory before parent
            // resumes its execution
            let barrier = Arc::new(Barrier::new(2));
            let barrier_clone = Arc::clone(&barrier);

            let builder = thread::Builder::new().name(format!("lind-fork-{}", child_pid));
            builder.spawn(move || {
                // create a new instance
                let mut store = Store::new(&instance_pre.module().engine(), child_host);
                let instance = instance_pre.instantiate(&mut store).unwrap();

                // copy the entire memory from parent, note that the unwind data is also copied together
                // with the memory
                let mut child_address = 0 as *mut u8;
                let mut address_length: usize = 0;

                for instance_item in &mut store.inner_mut().instances {
                    let vm_instance = instance_item.handle.instance_mut();

                    let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
                    child_address = defined_memory.base;
                    address_length = defined_memory.current_length.load(Ordering::SeqCst);
                    break;
                }

                if child_address == 0 as *mut u8 {
                    println!("no memory found for child");
                    return -1;
                }

                unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, child_address, address_length); }
        
                barrier_clone.wait();

                // create the cage in rustposix via rustposix fork
                lind_syscall_inner(parent_cageid, 68, 0, 0, child_cageid, 0, 0, 0, 0, 0);

                // get the asyncify_rewind_start and module start function
                let child_rewind_start;
                let child_start_func = instance
                            .get_func(&mut store, "_start")
                            .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();

                match instance.get_typed_func::<i32, ()>(&mut store, ASYNCIFY_START_REWIND) {
                    Ok(func) => {
                        child_rewind_start = func;
                    },
                    Err(error) => {
                        return -1;
                    }
                };

                // rewind the child
                let _ = child_rewind_start.call(&mut store, (rewind_pointer as i32));

                // set up rewind state and fork return value for child
                store.as_context_mut().0.rewinding = RewindingReturn {
                    rewinding: true,
                    retval: 0,
                };
                
                let ty = child_start_func.ty(&store);
                        
                let values = Vec::new();
                let mut results = vec![Val::null_func_ref(); ty.results().len()];

                let invoke_res = child_start_func
                    .call(&mut store, &values, &mut results);

                return 0;
            }).unwrap();

            barrier.wait();

            let _ = asyncify_start_rewind_func.call(&mut store, (rewind_pointer as i32));

            // set up rewind state and fork return value for parent
            store.0.inner.rewinding = RewindingReturn {
                rewinding: true,
                retval: child_pid,
            };;

            // return InvokeAgain here would make parent re-invoke main
            return Ok(OnCalledAction::InvokeAgain);
        }));

        return Ok(0);
    }

    fn next_process_id(&self) -> Option<i32> {
        match self
            .next_pid
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
                ..=0x1ffffffe => Some(v + 1),
                _ => None,
            }) {
            Ok(v) => Some(v + 1),
            Err(_) => None,
        }
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

    pub fn fork(&self) -> Self {
        let pid = match self.next_process_id() {
            Some(id) => id,
            None => {
                // to-do: should have a better error handling
                // instead of just panicking here
                panic!("running out of process id");
            }
        };

        let forked_ctx = Self {
            instance_pre: self.instance_pre.clone(),
            pid,
            next_pid: self.next_pid.clone(),
            next_cageid: self.next_cageid.clone()
        };

        return forked_ctx;
    }
}

pub fn add_to_linker<T: Clone + Send + 'static + std::marker::Sync>(
    linker: &mut wasmtime::Linker<T>,
    store: &wasmtime::Store<T>,
    module: &Module,
    get_cx: impl Fn(&T) -> &WasiForkCtx<T> + Send + Sync + Copy + 'static,
    get_wasi_cx: impl Fn(&mut T) -> &mut WasiCtx + Send + Sync + Copy + 'static,
    fork_host: impl Fn(&T) -> T + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    linker.func_wrap(
        "wasix",
        "lind-fork",
        move |mut caller: Caller<'_, T>, _: i32| -> i32 {
            let host = caller.data().clone();
            let ctx = get_cx(&host);

            match ctx.fork_call(&mut caller, get_cx, get_wasi_cx, fork_host) {
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
