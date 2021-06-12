extern crate plugin_derive;

pub use plugin_api as api;
pub use plugin_derive::{event_handler, global_state};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    borrow::{Cow, ToOwned},
    convert::TryInto,
    marker::PhantomData,
};

pub fn __game() -> api::Game { api::Game::__new(|_| todo!()) }

#[cfg(target_arch = "wasm32")]
extern "C" {
    fn raw_emit_actions(ptr: i64, len: i64);
    fn raw_request(ptr: i64, len: i64) -> i64;
    fn raw_print(ptr: i64, len: i64);
}

pub fn request(req: api::raw::RawRequest) -> Result<api::raw::RawResponse<'static>, ()> {
    #[cfg(target_arch = "wasm32")]
    {
        let ret = bincode::serialize(&req).expect("Can't serialize action in emit");
        let bytes = unsafe {
            let ptr = raw_request(to_i64(ret.as_ptr() as _), to_i64(ret.len() as _));
            let ptr = from_i64(ptr);
            let len =
                u64::from_le_bytes(std::slice::from_raw_parts(ptr as _, 8).try_into().unwrap());
            ::std::slice::from_raw_parts((ptr + 8) as *const u8, len as _)
        };
        bincode::deserialize::<Result<api::raw::RawResponse<'static>, ()>>(bytes).map_err(|_| ())?
    }
    #[cfg(not(target_arch = "wasm32"))]
    panic!("Requests are not implemented for non-WASM targets")
}

pub fn emit_action(action: api::raw::RawAction) { emit_actions(&[action]) }

pub fn emit_actions(actions: &[api::raw::RawAction]) {
    #[cfg(target_arch = "wasm32")]
    {
        let ret = bincode::serialize(actions.as_ref()).expect("Can't serialize action in emit");
        unsafe {
            raw_emit_actions(to_i64(ret.as_ptr() as _), to_i64(ret.len() as _));
        }
    }
}

pub fn print_str(s: &str) {
    let bytes = s.as_bytes();
    unsafe {
        // Safety: ptr and len are valid for byte slice
        raw_print(to_i64(bytes.as_ptr() as _), to_i64(bytes.len() as _));
    }
}

#[macro_export]
macro_rules! log {
    ($($x:tt)*) => { $crate::print_str(&format!($($x)*)) };
}

/// Safety: Data pointed to by `ptr` must outlive 'a
pub unsafe fn read_input<'a, T>(ptr: i64, len: i64) -> Result<T, &'static str>
where
    T: Deserialize<'a>,
{
    let slice = unsafe { ::std::slice::from_raw_parts(from_i64(ptr) as _, from_i64(len) as _) };
    bincode::deserialize(slice).map_err(|_| "Failed to deserialize function input")
}

/// This function split a u128 in two u64 encoding them as le bytes
pub fn from_u128(i: u128) -> (u64, u64) {
    let i = i.to_le_bytes();
    (
        u64::from_le_bytes(i[0..8].try_into().unwrap()),
        u64::from_le_bytes(i[8..16].try_into().unwrap()),
    )
}

/// This function merge two u64 encoded as le in one u128
pub fn to_u128(a: u64, b: u64) -> u128 {
    let a = a.to_le_bytes();
    let b = b.to_le_bytes();
    u128::from_le_bytes([a, b].concat().try_into().unwrap())
}

/// This function encode a u64 into a i64 using le bytes
pub fn to_i64(i: u64) -> i64 { i64::from_le_bytes(i.to_le_bytes()) }

/// This function decode a i64 into a u64 using le bytes
pub fn from_i64(i: i64) -> u64 { u64::from_le_bytes(i.to_le_bytes()) }

static mut VEC: Vec<u8> = vec![];
static mut DATA: Vec<u8> = vec![];

pub fn write_output(value: impl Serialize) -> i64 {
    unsafe {
        VEC = bincode::serialize(&value).expect("Can't serialize event output");
        DATA = [
            (VEC.as_ptr() as u64).to_le_bytes(),
            (VEC.len() as u64).to_le_bytes(),
        ]
        .concat();
        to_i64(DATA.as_ptr() as u64)
    }
}

static mut BUFFERS: Vec<u8> = Vec::new();

/// Allocate buffer from wasm linear memory
/// # Safety
/// This function should never be used only intented to by used by the host
#[no_mangle]
pub unsafe fn wasm_prepare_buffer(size: i32) -> i64 {
    BUFFERS = vec![0u8; size as usize];
    BUFFERS.as_ptr() as i64
}
