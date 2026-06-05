//! Packing API and PIR-local BSGS DFT-hot implementation.
//!
//! Public callers use [`Packing`] with two separated inputs:
//! fixed mask-side [`PackingPrecomputations`] and client-key-side
//! [`PackingKeys`]. The remaining submodules are implementation
//! layers for allocation/precompute, the online hot path, and backend OEP
//! wiring.

mod api;
mod default;
mod delegates;
mod oep;
#[allow(clippy::module_inception)]
mod packing;
mod packing_keys;
mod packing_mask_preprocessing;
mod packing_precomputations;
pub(crate) mod recursion;

pub use api::*;
pub use packing_keys::*;
pub use packing_precomputations::*;

#[cfg(test)]
mod tests;
