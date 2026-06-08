//! The database side of the toy PIR.
//!
//! A single [`DatabaseLayout`] describes the coefficient matrix shape for **both**
//! constructions: total coefficient `rows × cols`.
//! Scheme-specific quantities are derived outside the layout:
//! - **InsPIRe** uses `column_height = n`, so `block_rows = rows / n`.
//! - **InsPIRe²** gets `γ0` from `Collapse::Recursion`, so `t = rows / γ0`.
//!
//! Payloads are generic via [`Payload`]: a payload (here `[u8; 32]`) packs into
//! `P::EXPONENT` consecutive coefficient rows of one column, each a centred-`i16`
//! base-`P::BASIS` digit. [`Database`] holds the InsPIRe `n × n` `i16` matrices;
//! `base2k` is a coefficient-storage detail supplied at [`DatabaseLayout::instantiate`].
//!
//! [`Payload`]: crate::payload::Payload

mod address;
mod coeff_matrix;
mod layout;
mod preprocessing;
mod storage;

pub use address::{Address, PayloadAddress};
pub use coeff_matrix::CoeffMatrix;
pub use layout::DatabaseLayout;
pub use preprocessing::DatabasePreprocessingConfig;
pub use storage::Database;
