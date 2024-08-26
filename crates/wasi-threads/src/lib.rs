//! Implement [`wasi-threads`].
//!
//! [`wasi-threads`]: https://github.com/WebAssembly/wasi-threads

use anyhow::{anyhow, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Barrier};
use std::thread;
use wasmtime::{AsContext, Caller, ExternType, Func, InstancePre, Linker, Module, SharedMemory, Store, StoreContextMut, Val};

use wasmtime::runtime::vm::vmcontext::VMContext;
use wasmtime::runtime::vm::VMMemoryImport;
use wasmtime::runtime::vm::Instance;
use wasmtime::runtime::vm::VMRuntimeLimits;

use wasmtime_environ::MemoryIndex;

// This name is a function export designated by the wasi-threads specification:
// https://github.com/WebAssembly/wasi-threads/#detailed-design-discussion
const WASI_ENTRY_POINT: &str = "wasi_thread_start";

struct LocalRuntimeLimit {
    pub stack_limit: usize,

    pub fuel_consumed: i64,

    pub epoch_deadline: u64,

    pub last_wasm_exit_fp: usize,

    pub last_wasm_exit_pc: usize,

    pub last_wasm_entry_sp: usize,
}

pub struct WasiThreadsCtx<T> {
    instance_pre: Arc<InstancePre<T>>,
    tid: AtomicI32,
}

impl<T: Clone + Send + 'static> WasiThreadsCtx<T> {
    pub fn new(module: Module, linker: Arc<Linker<T>>) -> Result<Self> {
        let instance_pre = Arc::new(linker.instantiate_pre(&module)?);
        let tid = AtomicI32::new(0);
        Ok(Self { instance_pre, tid })
    }

    pub fn spawn(&self, host: T, thread_start_arg: i32) -> Result<i32> {
        let instance_pre = self.instance_pre.clone();

        // Check that the thread entry point is present. Why here? If we check
        // for this too early, then we cannot accept modules that do not have an
        // entry point but never spawn a thread. As pointed out in
        // https://github.com/bytecodealliance/wasmtime/issues/6153, checking
        // the entry point here allows wasi-threads to be compatible with more
        // modules.
        //
        // As defined in the wasi-threads specification, returning a negative
        // result here indicates to the guest module that the spawn failed.
        if !has_entry_point(instance_pre.module()) {
            log::error!("failed to find a wasi-threads entry point function; expected an export with name: {WASI_ENTRY_POINT}");
            return Ok(-1);
        }
        if !has_correct_signature(instance_pre.module()) {
            log::error!("the exported entry point function has an incorrect signature: expected `(i32, i32) -> ()`");
            return Ok(-1);
        }

        let wasi_thread_id = self.next_thread_id();
        if wasi_thread_id.is_none() {
            log::error!("ran out of valid thread IDs");
            return Ok(-1);
        }
        let wasi_thread_id = wasi_thread_id.unwrap();

        // Start a Rust thread running a new instance of the current module.
        let builder = thread::Builder::new().name(format!("wasi-thread-{}", wasi_thread_id));
        builder.spawn(move || {
            // Catch any panic failures in host code; e.g., if a WASI module
            // were to crash, we want all threads to exit, not just this one.
            let result = catch_unwind(AssertUnwindSafe(|| {
                // Each new instance is created in its own store.
                let mut store = Store::new(&instance_pre.module().engine(), host);
                // println!("-----------------thread memory------------------");
                let instance = instance_pre.instantiate(&mut store).unwrap();
                let thread_entry_point = instance
                    .get_typed_func::<(i32, i32), ()>(&mut store, WASI_ENTRY_POINT)
                    .unwrap();

                // Start the thread's entry point. Any traps or calls to
                // `proc_exit`, by specification, should end execution for all
                // threads. This code uses `process::exit` to do so, which is
                // what the user expects from the CLI but probably not in a
                // Wasmtime embedding.
                log::trace!(
                    "spawned thread id = {}; calling start function `{}` with: {}",
                    wasi_thread_id,
                    WASI_ENTRY_POINT,
                    thread_start_arg
                );

                // println!(
                //     "spawned thread id = {}; calling start function `{}` with: {}",
                //     wasi_thread_id,
                //     WASI_ENTRY_POINT,
                //     thread_start_arg
                // );

                match thread_entry_point.call(&mut store, (wasi_thread_id, thread_start_arg)) {
                    Ok(_) => { 
                        log::trace!("exiting thread id = {} normally", wasi_thread_id);
                        // println!("exiting thread id = {} normally", wasi_thread_id);
                    }
                    Err(e) => {
                        log::trace!("exiting thread id = {} due to error", wasi_thread_id);
                        let e = wasi_common::maybe_exit_on_error(e);
                        eprintln!("Error: {:?}", e);
                        std::process::exit(1);
                    }
                }
                // println!("wasi-thread-spawn done");
            }));

            if let Err(e) = result {
                eprintln!("wasi-thread-{} panicked: {:?}", wasi_thread_id, e);
                std::process::exit(1);
            }
        })?;

        Ok(wasi_thread_id)
    }

    pub fn fork(&self, host: T, parent_address: *mut u8, parent_runtime_limit: LocalRuntimeLimit) -> Result<i32> {
        println!("inside fork!");

        let instance_pre = self.instance_pre.clone();

        let wasi_thread_id = self.next_thread_id();
        if wasi_thread_id.is_none() {
            log::error!("ran out of valid thread IDs");
            return Ok(-1);
        }
        let wasi_thread_id = wasi_thread_id.unwrap();
        if wasi_thread_id > 1
        {
            return Ok(1);
        }

        // Start a Rust thread running a new instance of the current module.
        let builder = thread::Builder::new().name(format!("wasi-thread-{}", wasi_thread_id));

        let module = instance_pre.module();
        let offsets = module.offsets();
        let vm_offset = offsets.vmctx_imported_memories_begin();
        // let vm_offset = offsets.vmctx_vmmemory_import(MemoryIndex);

        // println!("vmoffsets: {:?}", offsets);

        // unsafe {
        //     let tmp = (host_vmctx as *const u8);
        //     println!("host_vmctx: {:?}", host_vmctx);

        //     let imported_memory_ptr = (host_vmctx as *const u8).offset(vm_offset as isize);
        //     println!("imported_memory_ptr: {:?}", imported_memory_ptr);
        //     let memories_ptr = imported_memory_ptr as *const *const VMMemoryImport;

        //     let first_memory = *memories_ptr;

        //     if !first_memory.is_null() {
        //         // Dereference the pointer and print out the memory details
        //         println!("Imported memory details: {:?}", *first_memory);
        //     } else {
        //         println!("No imported memories found.");
        //     }
        // }

        // unsafe { println!("vmctx: {:?}", *host_vmctx); }

        // let host_vmctx_int: usize = host_vmctx as usize;
        let parent_address_int: usize = parent_address as usize;

        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);

        builder.spawn(move || {
            // Catch any panic failures in host code; e.g., if a WASI module
            // were to crash, we want all threads to exit, not just this one.
            let result = catch_unwind(AssertUnwindSafe(|| {
                // Each new instance is created in its own store.
                let mut store = Store::new(&instance_pre.module().engine(), host);
                let instance = instance_pre.instantiate(&mut store).unwrap();


                let runtime_limits = &store.as_context().0.runtime_limits();

                // unsafe { *runtime_limits.stack_limit.get() = parent_runtime_limit.stack_limit };
                // unsafe { *runtime_limits.fuel_consumed.get() = parent_runtime_limit.fuel_consumed };
                unsafe { *runtime_limits.last_wasm_exit_fp.get() = parent_runtime_limit.last_wasm_exit_fp };
                // unsafe { *runtime_limits.epoch_deadline.get() = parent_runtime_limit.epoch_deadline };
                unsafe { *runtime_limits.last_wasm_exit_pc.get() = parent_runtime_limit.last_wasm_exit_pc };
                unsafe { *runtime_limits.last_wasm_entry_sp.get() = parent_runtime_limit.last_wasm_entry_sp };
                
                let stack_limit: usize = unsafe { *runtime_limits.stack_limit.get() };
                let fuel_consumed: i64 = unsafe { *runtime_limits.fuel_consumed.get() };
                let fp: usize = unsafe { *runtime_limits.last_wasm_exit_fp.get() };
                let epoch_deadline: u64 = unsafe { *runtime_limits.epoch_deadline.get() };
                let last_wasm_exit_pc: usize = unsafe { *runtime_limits.last_wasm_exit_pc.get() };
                let last_wasm_entry_sp: usize = unsafe { *runtime_limits.last_wasm_entry_sp.get() };
                println!("child stack_limit: {:?}", stack_limit);
                println!("child fuel_consumed: {:?}", fuel_consumed);
                println!("child last_wasm_exit_fp: {:?}", fp);
                println!("child epoch_deadline: {:?}", epoch_deadline);
                println!("child last_wasm_exit_pc: {:?}", last_wasm_exit_pc);
                println!("child last_wasm_entry_sp: {:?}", last_wasm_entry_sp);

                let parent_address_back: *mut u8 = parent_address_int as *mut u8;
                // let instances = &store.inner().instances;
                for instance_item in &mut store.inner_mut().instances {
                    let vm_instance = instance_item.handle.instance_mut();

                    let runtime_limits_ins;
                    unsafe { runtime_limits_ins = &**vm_instance.runtime_limits(); };

                    let stack_limit: usize = unsafe { *runtime_limits_ins.stack_limit.get() };
                    let fuel_consumed: i64 = unsafe { *runtime_limits_ins.fuel_consumed.get() };
                    let fp: usize = unsafe { *runtime_limits_ins.last_wasm_exit_fp.get() };
                    let epoch_deadline: u64 = unsafe { *runtime_limits_ins.epoch_deadline.get() };
                    let last_wasm_exit_pc: usize = unsafe { *runtime_limits_ins.last_wasm_exit_pc.get() };
                    let last_wasm_entry_sp: usize = unsafe { *runtime_limits_ins.last_wasm_entry_sp.get() };
                    println!("child_ins stack_limit: {:?}", stack_limit);
                    println!("child_ins fuel_consumed: {:?}", fuel_consumed);
                    println!("child_ins last_wasm_exit_fp: {:?}", fp);
                    println!("child_ins epoch_deadline: {:?}", epoch_deadline);
                    println!("child_ins last_wasm_exit_pc: {:?}", last_wasm_exit_pc);
                    println!("child_ins last_wasm_entry_sp: {:?}", last_wasm_entry_sp);

                    println!("child offset: {:?}", vm_instance.offsets());
                    let globals = vm_instance.defined_globals();
                    println!("child globals");
                    for (index, global) in globals {
                        unsafe {
                            if index.0 == 0 {
                                let num: u128 = 131424;
                                (*global.definition).storage = num.to_le_bytes();
                            }
                            let res = u128::from_le_bytes((*global.definition).storage);
                            println!("{:?}: {:?}, {:?}", index, res, global.global);
                        }
                    }
                    let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
                    println!("child defined_memory: {:?}", defined_memory);
                    let length: usize = defined_memory.current_length.load(Ordering::SeqCst);
                    unsafe { std::ptr::copy_nonoverlapping(parent_address_back, defined_memory.base, length); }
                }
                barrier_clone.wait();

                // instance.print_externs();

                // may need to be replaced by something like self.invoke in run.rs
                let func = instance
                            .get_func(&mut store, "_start")
                            .ok_or_else(|| anyhow!("no func export named `_start` found")).unwrap();
                
                let ty = func.ty(&store);
                        
                let mut values = Vec::new();
                let mut results = vec![Val::null_func_ref(); ty.results().len()];

                // need to handle rustposix fork here
                
                // let host_vmctx_back: *mut VMContext = host_vmctx_int as *mut VMContext;

                // let invoke_res = func
                //     .call_fork(&mut store, &values, &mut results, host_vmctx_back);
                let invoke_res = func
                    .call(&mut store, &values, &mut results);
                println!("result: {:?}", invoke_res);

                // println!(
                //     "spawned thread id = {}; calling start function `{}` with: {}",
                //     wasi_thread_id,
                //     WASI_ENTRY_POINT,
                //     thread_start_arg
                // );
                // println!("wasi-thread-spawn done");
            }));

            // if let Err(e) = result {
            //     eprintln!("wasi-thread-{} panicked: {:?}", wasi_thread_id, e);
            //     std::process::exit(1);
            // }
        })?;

        barrier.wait();
        Ok(1)
    }

    /// Helper for generating valid WASI thread IDs (TID).
    ///
    /// Callers of `wasi_thread_spawn` expect a TID in range of 0 < TID <= 0x1FFFFFFF
    /// to indicate a successful spawning of the thread whereas a negative
    /// return value indicates an failure to spawn.
    fn next_thread_id(&self) -> Option<i32> {
        match self
            .tid
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
pub fn add_to_linker<T: Clone + Send + 'static>(
    linker: &mut wasmtime::Linker<T>,
    store: &wasmtime::Store<T>,
    module: &Module,
    get_cx: impl Fn(&mut T) -> &WasiThreadsCtx<T> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    linker.func_wrap(
        "wasi",
        "thread-spawn",
        move |mut caller: Caller<'_, T>, start_arg: i32| -> i32 {
            // println!("---------------------thread-spawn---------------------");
            log::trace!("new thread requested via `wasi::thread_spawn` call");
            let host = caller.data().clone();
            let ctx = get_cx(caller.data_mut());
            match ctx.spawn(host, start_arg) {
                Ok(thread_id) => {
                    assert!(thread_id >= 0, "thread_id = {}", thread_id);
                    thread_id
                }
                Err(e) => {
                    log::error!("failed to spawn thread: {}", e);
                    -1
                }
            }
        },
    )?;

    linker.func_wrap(
        "wasi",
        "wasi-fork",
        move |mut caller: Caller<'_, T>, start_arg: i32| -> i32 {
            println!("---------------------wasi-fork---------------------");
            let host = caller.data().clone();

            let runtime_limits = &caller.store.as_context().0.runtime_limits();

            let stack_limit: usize = unsafe { *runtime_limits.stack_limit.get() };
            let fuel_consumed: i64 = unsafe { *runtime_limits.fuel_consumed.get() };
            let fp: usize = unsafe { *runtime_limits.last_wasm_exit_fp.get() };
            let epoch_deadline: u64 = unsafe { *runtime_limits.epoch_deadline.get() };
            let last_wasm_exit_pc: usize = unsafe { *runtime_limits.last_wasm_exit_pc.get() };
            let last_wasm_entry_sp: usize = unsafe { *runtime_limits.last_wasm_entry_sp.get() };
            println!("parent stack_limit: {:?}", stack_limit);
            println!("parent fuel_consumed: {:?}", fuel_consumed);
            println!("parent last_wasm_exit_fp: {:?}", fp);
            println!("parent epoch_deadline: {:?}", epoch_deadline);
            println!("parent last_wasm_exit_pc: {:?}", last_wasm_exit_pc);
            println!("parent last_wasm_entry_sp: {:?}", last_wasm_entry_sp);

            // let fp = &runtime_limits.last_wasm_exit_fp;
            // let value: usize = unsafe { *fp.get() };
            // println!("parent last_wasm_exit_fp: {:?}", value);
            let tmp = &caller.store.as_context().0.instances.get(0);
            let vmctx;
            let vm_instance;
            let address;
            if let Some(instance_item) = tmp {
                vm_instance = instance_item.handle.instance();
                let defined_memory = vm_instance.get_memory(MemoryIndex::from_u32(0));
                println!("parent offset: {:?}", vm_instance.offsets());
                let globals = vm_instance.defined_globals_immut();
                println!("parent globals");
                for (index, global) in globals {
                    unsafe {
                        let res = u128::from_le_bytes((*global.definition).storage);
                        println!("{:?}: {:?}, {:?}", index, res, global.global);
                    }
                }
                println!("parent memory: {:?}", defined_memory);
                address = defined_memory.base;
                // let module_info = vm_instance.runtime_info;
                // if let Module(module) = module_info {

                // }
                vmctx = vm_instance.vmctx();
            } else {
                return -1;
            }
            let ctx = get_cx(caller.data_mut());

            match ctx.fork(host, address, LocalRuntimeLimit {
                stack_limit: stack_limit,
                fuel_consumed: fuel_consumed,
                epoch_deadline: epoch_deadline,
                last_wasm_exit_fp: fp,
                last_wasm_exit_pc: last_wasm_exit_pc,
                last_wasm_entry_sp: last_wasm_entry_sp,
            }) {
                Ok(pid) => {
                    // assert!(thread_id >= 0, "thread_id = {}", thread_id);
                    println!("pid={}", pid);
                    pid
                }
                Err(e) => {
                    log::error!("failed to fork: {}", e);
                    -1
                }
            }
        },
    )?;

    // Find the shared memory import and satisfy it with a newly-created shared
    // memory import.
    // println!("wasi-thread iterate import");
    for import in module.imports() {
        // println!("an import here");
        // println!("ty: {:?}", import.ty());
        if let Some(m) = import.ty().memory() {
            // println!("has memory");
            if m.is_shared() {
                // println!("wasi-thread share memory");
                let mem = SharedMemory::new(module.engine(), m.clone())?;
                linker.define(store, import.module(), import.name(), mem.clone())?;
            } else {
                return Err(anyhow!(
                    "memory was not shared; a `wasi-threads` must import \
                     a shared memory as \"memory\""
                ));
            }
        }
    }
    Ok(())
}

/// Check if wasi-threads' `wasi_thread_start` export is present.
fn has_entry_point(module: &Module) -> bool {
    module.get_export(WASI_ENTRY_POINT).is_some()
}

/// Check if the entry function has the correct signature `(i32, i32) -> ()`.
fn has_correct_signature(module: &Module) -> bool {
    match module.get_export(WASI_ENTRY_POINT) {
        Some(ExternType::Func(ty)) => {
            ty.params().len() == 2
                && ty.params().nth(0).unwrap().is_i32()
                && ty.params().nth(1).unwrap().is_i32()
                && ty.results().len() == 0
        }
        _ => false,
    }
}
