use anyhow::{anyhow, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Barrier};
use std::thread;
use wasmtime::{AsContext, AsContextMut, Caller, ExternType, Func, InstancePre, Linker, Module, SharedMemory, Store, StoreContextMut, TypedFunc, Val};

use wasmtime::runtime::vm::vmcontext::VMContext;
use wasmtime::runtime::vm::VMMemoryImport;
use wasmtime::runtime::vm::Instance;
use wasmtime::runtime::vm::VMRuntimeLimits;
use wasmtime::runtime::Extern;
use wasmtime::runtime::OnCalledAction;
use wasmtime::runtime::RewindingReturn;

use wasmtime_environ::MemoryIndex;

pub struct WasiForkCtx<T> {
    instance_pre: Arc<InstancePre<T>>,
    pid: AtomicI32,
}

impl<T: Clone + Send + 'static> WasiForkCtx<T> {
    pub fn new(module: Module, linker: Arc<Linker<T>>) -> Result<Self> {
        let instance_pre = Arc::new(linker.instantiate_pre(&module)?);
        let pid = AtomicI32::new(0);
        Ok(Self { instance_pre, pid })
    }

    pub fn fork(&self, host: T, caller: Caller<'_, T>) -> Result<i32> {
        Ok(1)
    }

    pub fn next_process_id(&self) -> Option<i32> {
        match self
            .pid
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| match v {
                ..=0x1ffffffe => Some(v + 1),
                _ => None,
            }) {
            Ok(v) => Some(v + 1),
            Err(_) => None,
        }
    }
}

/// Manually add the WASI `thread_spawn` function to the linker.
///
/// It is unclear what namespace the `wasi-threads` proposal should live under:
/// it is not clear if it should be included in any of the `preview*` releases
/// so for the time being its module namespace is simply `"wasi"` (TODO).
pub fn add_to_linker<T: Clone + Send + 'static + std::marker::Sync>(
    linker: &mut wasmtime::Linker<T>,
    store: &wasmtime::Store<T>,
    module: &Module,
    get_cx: impl Fn(&mut T) -> &WasiForkCtx<T> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    println!("lind fork add to linker");
    linker.func_wrap(
        "wasix",
        "lind-fork",
        move |mut caller: Caller<'_, T>, _: i32| -> i32 {
            // _println!("---------------------wasi-fork---------------------");
            let host = caller.data().clone();

            if caller.store.0.inner.rewinding.rewinding {
                if let Some(asyncify_stop_rewind_extern) = caller.get_export("asyncify_stop_rewind") {
                    match asyncify_stop_rewind_extern {
                        Extern::Func(asyncify_stop_rewind) => {
                            match asyncify_stop_rewind.typed::<(), ()>(&caller) {
                                Ok(func) => {
                                    func.call(&mut caller, ());
                                }
                                Err(err) => {
                                    println!("something went wrong");
                                    return -1;
                                }
                            }
                        },
                        _ => {
                            println!("asyncify_stop_rewind export is not a function");
                            return -1;
                        }
                    }
                }
                else {
                    println!("asyncify_stop_rewind export not found");
                    return -1;
                }

                let retval = caller.store.0.inner.rewinding.retval;

                caller.store.0.inner.rewinding = RewindingReturn {
                    rewinding: false,
                    retval: 0,
                };

                return retval;
            }

            let tmp = &caller.store.as_context().0.instances.get(0);
            let vmctx;
            let vm_instance;
            let address;
            if let Some(instance_item) = tmp {
                vm_instance = instance_item.handle.instance();
                let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
                // _println!("parent offset: {:?}", vm_instance.offsets());
                let globals = vm_instance.defined_globals_immut();
                // _println!("parent globals");
                for (index, global) in globals {
                    unsafe {
                        let res = u128::from_le_bytes((*global.definition).storage);
                        // _println!("{:?}: {:?}, {:?}", index, res, global.global);
                    }
                }
                // _println!("parent memory: {:?}", defined_memory);
                address = defined_memory.base;
                // let module_info = vm_instance.runtime_info;
                // if let Module(module) = module_info {

                // }
                vmctx = vm_instance.vmctx();
            } else {
                return -1;
            }

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
                                return -1;
                            }
                        }
                    },
                    _ => {
                        println!("__stack_pointer export is not a Global");
                        return -1;
                    }
                }
            }
            else {
                println!("__stack_pointer export not found");
                return -1;
            }

            // _println!("stack_pointer: {:?}", stack_pointer);

            if let Some(asyncify_start_unwind_extern) = caller.get_export("asyncify_start_unwind") {
                match asyncify_start_unwind_extern {
                    Extern::Func(asyncify_start_unwind) => {
                        match asyncify_start_unwind.typed::<i32, ()>(&caller) {
                            Ok(func) => {
                                let unwind_pointer: u64 = 0;
                                let unwind_data_start: u64 = unwind_pointer + 8;
                                let unwind_data_end: u64 = stack_pointer as u64;
        
                                unsafe {
                                    *(address as *mut u64) = unwind_data_start;
                                    *(address as *mut u64).add(1) = unwind_data_end;
                                    // _println!("before: unwind_data_start: {}, unwind_data_end: {}", *(address as *mut u64), *(address as *mut u64).add(1));
                                }
        
                                func.call(&mut caller, (unwind_pointer as i32));
                            }
                            Err(err) => {
                                println!("something went wrong");
                                return -1;
                            }
                        }
                    },
                    _ => {
                        println!("asyncify_start_unwind export is not a function");
                        return -1;
                    }
                }
            }
            else {
                println!("asyncify_start_unwind export not found");
                return -1;
            }

            let asyncify_stop_unwind_func;

            if let Some(asyncify_stop_unwind_extern) = caller.get_export("asyncify_stop_unwind") {
                match asyncify_stop_unwind_extern {
                    Extern::Func(asyncify_stop_unwind) => {
                        match asyncify_stop_unwind.typed::<(), ()>(&caller) {
                            Ok(func) => {
                                asyncify_stop_unwind_func = func;
                            }
                            Err(err) => {
                                println!("something went wrong");
                                return -1;
                            }
                        }
                    },
                    _ => {
                        println!("asyncify_stop_unwind export is not a function");
                        return -1;
                    }
                }
            }
            else {
                println!("asyncify_stop_unwind export not found");
                return -1;
            }

            let asyncify_start_rewind_func;

            if let Some(asyncify_start_rewind_extern) = caller.get_export("asyncify_start_rewind") {
                match asyncify_start_rewind_extern {
                    Extern::Func(asyncify_start_rewind) => {
                        match asyncify_start_rewind.typed::<i32, ()>(&caller) {
                            Ok(func) => {
                                asyncify_start_rewind_func = func;
                            }
                            Err(err) => {
                                println!("something went wrong");
                                return -1;
                            }
                        }
                    },
                    _ => {
                        println!("asyncify_start_rewind export is not a function");
                        return -1;
                    }
                }
            }
            else {
                println!("asyncify_start_rewind export not found");
                return -1;
            }

            let cloned_address = address as u64;

            let ctx = get_cx(caller.data_mut());
            let instance_pre = ctx.instance_pre.clone();
            let child_pid = ctx.next_process_id();

            if let None = child_pid {
                println!("running out of pid");
                return -1;
            }
            let child_pid = child_pid.unwrap();
            
            let store = caller.store.0;

            store.set_on_called(Box::new(move |mut store| {
                // _println!("on called!");
                let unwind_stack_finish;

                let address = cloned_address as *mut u64;
                let unwind_start_address = (cloned_address + 8) as *mut u64;

                unsafe {
                    unwind_stack_finish = *address;
                }

                let unwind_size = unwind_stack_finish - 8;
                let mut unwind_stack = Vec::with_capacity(unwind_size as usize);

                unsafe {
                    // Create a slice from the source pointer
                    let src_slice = std::slice::from_raw_parts(unwind_start_address as *mut u8, unwind_size as usize);
                    // Extend the buffer with the slice
                    unwind_stack.extend_from_slice(src_slice);
                }

                // _println!("unwind_stack: {:?}", unwind_stack);

                // if let Some(ins) = store.0.inner.default_caller.instance().host_state().downcast_ref::<wasmtime::runtime::Instance>() {
                //     let asyncify_stop_unwind = ins.get_export(&mut store, "asyncify_stop_unwind");
                // }
                // else {
                //     println!("asyncify_start_unwind export not found");
                //     return Ok(OnCalledAction::Finish);
                // }
                asyncify_stop_unwind_func.call(&mut store, ());

                let rewind_pointer: u64 = 0;
                // let rewind_data_start: u64 = rewind_pointer + 8;
                // let rewind_data_end: u64 = stack_pointer as u64;

                // unsafe {
                    // *(address as *mut u64) = unwind_stack_finish;
                    // *(address as *mut u64).add(1) = rewind_data_end;
                //     println!("before: rewind_data_start: {}, rewind_data_end: {}", *(address as *mut u64), *(address as *mut u64).add(1));
                // }

                // unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, cloned_child_address as *mut u8, address_length); }

                let barrier = Arc::new(Barrier::new(2));
                let barrier_clone = Arc::clone(&barrier);

                let builder = thread::Builder::new().name(format!("wasi-thread-{}", 1));
                builder.spawn(move || {
                    // child_start_closure(rewind_pointer as i32);

                    // let instance_pre = ctx.instance_pre.clone();

                    let mut store = Store::new(&instance_pre.module().engine(), host);
                    let instance = instance_pre.instantiate(&mut store).unwrap();

                    let mut child_address = 0 as *mut u8;
                    let mut address_length: usize = 0;

                    for instance_item in &mut store.inner_mut().instances {
                        let vm_instance = instance_item.handle.instance_mut();

                        let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
                        child_address = defined_memory.base;
                        // _println!("child defined_memory: {:?}", defined_memory);
                        address_length = defined_memory.current_length.load(Ordering::SeqCst);
                        break;
                    }

                    if child_address == 0 as *mut u8 {
                        println!("no memory found for child");
                        return -1;
                    }

                    unsafe { std::ptr::copy_nonoverlapping(cloned_address as *mut u8, child_address, address_length); }
            
                    barrier_clone.wait();

                    let child_start_func = instance
                                .get_func(&mut store, "_start")
                                .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();
                    let child_rewind_start;

                    match instance.get_typed_func::<i32, ()>(&mut store, "asyncify_start_rewind") {
                        Ok(func) => {
                            child_rewind_start = func;
                        },
                        Err(error) => {
                            return -1;
                        }
                    };

                    child_rewind_start.call(&mut store, (rewind_pointer as i32));

                    store.as_context_mut().0.rewinding = RewindingReturn {
                        rewinding: true,
                        retval: child_pid,
                    };
                    
                    let ty = child_start_func.ty(&store);
                            
                    let mut values = Vec::new();
                    let mut results = vec![Val::null_func_ref(); ty.results().len()];

                    let invoke_res = child_start_func
                        .call(&mut store, &values, &mut results);
                    println!("child result: {:?}", invoke_res);

                    return 0;
                });

                barrier.wait();

                asyncify_start_rewind_func.call(&mut store, (rewind_pointer as i32));

                store.0.inner.rewinding = RewindingReturn {
                    rewinding: true,
                    retval: 0,
                };;

                return Ok(OnCalledAction::InvokeAgain);
            }));

            return 0;

            // let ctx = get_cx(caller.data_mut());

            match ctx.fork(host, caller) {
                Ok(pid) => {
                    // println!("pid={}", pid);
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

// Check if wasi-threads' `wasi_thread_start` export is present.
// fn has_entry_point(module: &Module) -> bool {
//     module.get_export(WASI_ENTRY_POINT).is_some()
// }

// Check if the entry function has the correct signature `(i32, i32) -> ()`.
// fn has_correct_signature(module: &Module) -> bool {
//     match module.get_export(WASI_ENTRY_POINT) {
//         Some(ExternType::Func(ty)) => {
//             ty.params().len() == 2
//                 && ty.params().nth(0).unwrap().is_i32()
//                 && ty.params().nth(1).unwrap().is_i32()
//                 && ty.results().len() == 0
//         }
//         _ => false,
//     }
// }
