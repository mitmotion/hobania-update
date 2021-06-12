#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::{Diagnostic, Level, TokenStream};
use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, ItemFn, ItemStruct};

#[proc_macro_attribute]
pub fn global_state(_args: TokenStream, item: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(item as ItemStruct);
    let name = &parsed.ident;
    let out: proc_macro2::TokenStream = quote! {
        #parsed
        type PLUGIN_STATE_TYPE = #name;

        static mut PLUGIN_STATE: Option<PLUGIN_STATE_TYPE> = None;

        static PLUGIN_STATE_GUARD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    };
    out.into()
}

#[proc_macro_attribute]
pub fn event_handler(_args: TokenStream, item: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(item as ItemFn);
    let fn_body = parsed.block; // function body
    let sig = parsed.sig; // function signature
    let fn_name = sig.ident; // function name/identifier
    let fn_args = sig.inputs; // comma separated args
    let fn_return = sig.output; // comma separated args

    let out: proc_macro2::TokenStream = match fn_args.len() {
        2 => quote! {
            #[allow(clippy::unnecessary_wraps)]
            #[no_mangle]
            pub fn #fn_name(intern__ptr: i64, intern__len: i64) -> i64 {
                /// Safety: `input` is not permitted to escape this function.
                let input = unsafe { ::veloren_plugin_rt::read_input(intern__ptr as _,intern__len as _).unwrap() };
                #[inline]
                fn inner(#fn_args) #fn_return {
                    #fn_body
                }
                // Artificially force the event handler to be type-correct
                fn force_event<'a, E: ::veloren_plugin_rt::api::Event<'a>>(
                    game: &'a ::veloren_plugin_rt::api::Game,
                    raw_event: E::Raw,
                    inner: fn(&::veloren_plugin_rt::api::Game, E) -> E::Response,
                ) -> E::Response {
                    inner(
                        &game,
                        E::from_raw(&game, raw_event),
                    )
                }
                ::veloren_plugin_rt::write_output(&force_event(unsafe { &::veloren_plugin_rt::__game() }, input, inner))
            }
        },
        3 => quote! {
            #[allow(clippy::unnecessary_wraps)]
            #[no_mangle]
            pub fn #fn_name(intern__ptr: i64, intern__len: i64) -> i64 {
                /// Safety: `input` is not permitted to escape this function.
                let input = unsafe { ::veloren_plugin_rt::read_input(intern__ptr as _,intern__len as _).unwrap() };
                #[inline]
                fn inner(#fn_args) #fn_return {
                    #fn_body
                }
                // Artificially force the event handler to be type-correct
                fn force_event<'a, E: ::veloren_plugin_rt::api::Event<'a>>(
                    game: &'a ::veloren_plugin_rt::api::Game,
                    raw_event: E::Raw,
                    inner: fn(&::veloren_plugin_rt::api::Game, E, &mut PLUGIN_STATE_TYPE) -> E::Response,
                ) -> E::Response {
                    assert_eq!(PLUGIN_STATE_GUARD.swap(true, std::sync::atomic::Ordering::Acquire), false);
                    let out = inner(
                        &game,
                        E::from_raw(&game, raw_event),
                        unsafe { PLUGIN_STATE.get_or_insert_with(core::default::Default::default) },
                    );
                    PLUGIN_STATE_GUARD.store(false, std::sync::atomic::Ordering::Release);
                    out

                }
                ::veloren_plugin_rt::write_output(&force_event(unsafe { &::veloren_plugin_rt::__game() }, input, inner))
            }
        },
        n => {
            Diagnostic::spanned(
                fn_args.span().unwrap(),
                Level::Error,
                "Incorrect number of function parameters",
            )
            .emit();
            quote! {}
        },
    };
    out.into()
}
