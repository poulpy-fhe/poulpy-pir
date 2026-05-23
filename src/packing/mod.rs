//! Packing API and PIR-local BSGS DFT-hot implementation.
//!
//! Public callers use [`Packing`] with two separated inputs:
//! fixed mask-side [`PackingPrecomputations`] and client-key-side
//! [`PackingKeyPrecomputations`]. The remaining submodules are implementation
//! layers for allocation/precompute, the online hot path, and backend OEP
//! wiring.

mod api;
mod bsgs_pack;
mod collapse_precompute;
mod default;
mod delegates;
mod key_precompute;
mod oep;

pub use api::*;
pub use collapse_precompute::*;

#[cfg(test)]
mod tests;
