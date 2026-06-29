use std::marker::PhantomData;

use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module},
};

use crate::{
    database::DatabaseLayout,
    parameters::Parameters,
    payload::{Payload, U256P65535, U256P65536},
};

pub const DEFAULT_N: usize = 2048;
pub const DEFAULT_BASE2K: usize = 18;
pub const DEFAULT_K: usize = 54;

/// Ready-made 32-byte PIR parameterizations for power-of-two database sizes.
///
/// Each variant names both the construction and the logical database size. Use
/// [`Self::resolve`] when the construction is selected dynamically, or
/// [`Self::interpolation`] / [`Self::recursion`] when the caller already knows
/// the payload type it expects. InsPIRe² variants cover `gamma0/gamma2` widths
/// `16/16`, `32/32`, and `64/64`, all with `gamma1=1024`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultPirParameters32B {
    InspireInt1GiB,
    InspireInt2GiB,
    InspireInt4GiB,
    InspireInt8GiB,
    InspireInt16GiB,
    InspireInt32GiB,
    InspireRecGamma16_1GiB,
    InspireRecGamma16_2GiB,
    InspireRecGamma16_4GiB,
    InspireRecGamma16_8GiB,
    InspireRecGamma16_16GiB,
    InspireRecGamma16_32GiB,
    InspireRecGamma32_1GiB,
    InspireRecGamma32_2GiB,
    InspireRecGamma32_4GiB,
    InspireRecGamma32_8GiB,
    InspireRecGamma32_16GiB,
    InspireRecGamma32_32GiB,
    InspireRecGamma64_1GiB,
    InspireRecGamma64_2GiB,
    InspireRecGamma64_4GiB,
    InspireRecGamma64_8GiB,
    InspireRecGamma64_16GiB,
    InspireRecGamma64_32GiB,
}

/// A default 32-byte parameterization resolved to its concrete payload type.
#[derive(Clone, Copy)]
pub enum DefaultPirConfig32B {
    Interpolation(DefaultPirInterpolationParams32B),
    Recursion(DefaultPirRecursionParams32B),
}

#[derive(Clone, Copy)]
pub struct DefaultPirInterpolationParams32B {
    pub db_size_gib: usize,
    pub config: Config<[u8; 32], U256P65535>,
    pub layout: DatabaseLayout<U256P65535>,
}

#[derive(Clone, Copy)]
pub struct DefaultPirRecursionParams32B {
    pub db_size_gib: usize,
    pub gamma0: usize,
    pub gamma1: usize,
    pub gamma2: usize,
    pub config: Config<[u8; 32], U256P65536>,
    pub layout: DatabaseLayout<U256P65536>,
}

impl DefaultPirParameters32B {
    pub const ALL: [Self; 24] = [
        Self::InspireInt1GiB,
        Self::InspireInt2GiB,
        Self::InspireInt4GiB,
        Self::InspireInt8GiB,
        Self::InspireInt16GiB,
        Self::InspireInt32GiB,
        Self::InspireRecGamma16_1GiB,
        Self::InspireRecGamma16_2GiB,
        Self::InspireRecGamma16_4GiB,
        Self::InspireRecGamma16_8GiB,
        Self::InspireRecGamma16_16GiB,
        Self::InspireRecGamma16_32GiB,
        Self::InspireRecGamma32_1GiB,
        Self::InspireRecGamma32_2GiB,
        Self::InspireRecGamma32_4GiB,
        Self::InspireRecGamma32_8GiB,
        Self::InspireRecGamma32_16GiB,
        Self::InspireRecGamma32_32GiB,
        Self::InspireRecGamma64_1GiB,
        Self::InspireRecGamma64_2GiB,
        Self::InspireRecGamma64_4GiB,
        Self::InspireRecGamma64_8GiB,
        Self::InspireRecGamma64_16GiB,
        Self::InspireRecGamma64_32GiB,
    ];

    pub const INTERPOLATION: [Self; 6] = [
        Self::InspireInt1GiB,
        Self::InspireInt2GiB,
        Self::InspireInt4GiB,
        Self::InspireInt8GiB,
        Self::InspireInt16GiB,
        Self::InspireInt32GiB,
    ];

    pub const RECURSION: [Self; 18] = [
        Self::InspireRecGamma16_1GiB,
        Self::InspireRecGamma16_2GiB,
        Self::InspireRecGamma16_4GiB,
        Self::InspireRecGamma16_8GiB,
        Self::InspireRecGamma16_16GiB,
        Self::InspireRecGamma16_32GiB,
        Self::InspireRecGamma32_1GiB,
        Self::InspireRecGamma32_2GiB,
        Self::InspireRecGamma32_4GiB,
        Self::InspireRecGamma32_8GiB,
        Self::InspireRecGamma32_16GiB,
        Self::InspireRecGamma32_32GiB,
        Self::InspireRecGamma64_1GiB,
        Self::InspireRecGamma64_2GiB,
        Self::InspireRecGamma64_4GiB,
        Self::InspireRecGamma64_8GiB,
        Self::InspireRecGamma64_16GiB,
        Self::InspireRecGamma64_32GiB,
    ];

    pub const fn db_size_gib(self) -> usize {
        match self {
            Self::InspireInt1GiB
            | Self::InspireRecGamma16_1GiB
            | Self::InspireRecGamma32_1GiB
            | Self::InspireRecGamma64_1GiB => 1,
            Self::InspireInt2GiB
            | Self::InspireRecGamma16_2GiB
            | Self::InspireRecGamma32_2GiB
            | Self::InspireRecGamma64_2GiB => 2,
            Self::InspireInt4GiB
            | Self::InspireRecGamma16_4GiB
            | Self::InspireRecGamma32_4GiB
            | Self::InspireRecGamma64_4GiB => 4,
            Self::InspireInt8GiB
            | Self::InspireRecGamma16_8GiB
            | Self::InspireRecGamma32_8GiB
            | Self::InspireRecGamma64_8GiB => 8,
            Self::InspireInt16GiB
            | Self::InspireRecGamma16_16GiB
            | Self::InspireRecGamma32_16GiB
            | Self::InspireRecGamma64_16GiB => 16,
            Self::InspireInt32GiB
            | Self::InspireRecGamma16_32GiB
            | Self::InspireRecGamma32_32GiB
            | Self::InspireRecGamma64_32GiB => 32,
        }
    }

    pub const fn rows(self) -> usize {
        1 << 16
    }

    pub const fn cols(self) -> usize {
        (1 << 13) * self.db_size_gib()
    }

    pub const fn collapse(self) -> Collapse {
        match self {
            Self::InspireInt1GiB
            | Self::InspireInt2GiB
            | Self::InspireInt4GiB
            | Self::InspireInt8GiB
            | Self::InspireInt16GiB
            | Self::InspireInt32GiB => Collapse::Interpolation,
            Self::InspireRecGamma16_1GiB
            | Self::InspireRecGamma16_2GiB
            | Self::InspireRecGamma16_4GiB
            | Self::InspireRecGamma16_8GiB
            | Self::InspireRecGamma16_16GiB
            | Self::InspireRecGamma16_32GiB => Collapse::Recursion {
                gamma0: 16,
                gamma1: 1024,
                gamma2: 16,
            },
            Self::InspireRecGamma32_1GiB
            | Self::InspireRecGamma32_2GiB
            | Self::InspireRecGamma32_4GiB
            | Self::InspireRecGamma32_8GiB
            | Self::InspireRecGamma32_16GiB
            | Self::InspireRecGamma32_32GiB => Collapse::Recursion {
                gamma0: 32,
                gamma1: 1024,
                gamma2: 32,
            },
            Self::InspireRecGamma64_1GiB
            | Self::InspireRecGamma64_2GiB
            | Self::InspireRecGamma64_4GiB
            | Self::InspireRecGamma64_8GiB
            | Self::InspireRecGamma64_16GiB
            | Self::InspireRecGamma64_32GiB => Collapse::Recursion {
                gamma0: 64,
                gamma1: 1024,
                gamma2: 64,
            },
        }
    }

    pub const fn gamma0(self) -> Option<usize> {
        match self.collapse() {
            Collapse::Interpolation => None,
            Collapse::Recursion { gamma0, .. } => Some(gamma0),
        }
    }

    pub const fn gamma1(self) -> Option<usize> {
        match self.collapse() {
            Collapse::Interpolation => None,
            Collapse::Recursion { gamma1, .. } => Some(gamma1),
        }
    }

    pub const fn gamma2(self) -> Option<usize> {
        match self.collapse() {
            Collapse::Interpolation => None,
            Collapse::Recursion { gamma2, .. } => Some(gamma2),
        }
    }

    pub const fn query_kib(self) -> usize {
        match self.collapse() {
            Collapse::Interpolation => 84 + 56 * self.db_size_gib(),
            Collapse::Recursion { gamma0: 16, .. } => match self.db_size_gib() {
                1 => 80,
                2 => 132,
                4 => 238,
                8 => 450,
                16 => 874,
                32 => 1722,
                _ => unreachable!(),
            },
            Collapse::Recursion { gamma0: 32, .. } => 13 + 53 * self.db_size_gib(),
            Collapse::Recursion { gamma0: 64, .. } => 7 + 53 * self.db_size_gib(),
            Collapse::Recursion { .. } => unreachable!(),
        }
    }

    pub fn resolve(self) -> DefaultPirConfig32B {
        match self.collapse() {
            Collapse::Interpolation => {
                DefaultPirConfig32B::Interpolation(DefaultPirInterpolationParams32B {
                    db_size_gib: self.db_size_gib(),
                    config: Config {
                        n: DEFAULT_N,
                        base2k: DEFAULT_BASE2K,
                        k: DEFAULT_K,
                        collapse: Collapse::Interpolation,
                        _phantom: PhantomData,
                    },
                    layout: DatabaseLayout::new(self.rows(), self.cols()),
                })
            }
            Collapse::Recursion {
                gamma0,
                gamma1,
                gamma2,
            } => DefaultPirConfig32B::Recursion(DefaultPirRecursionParams32B {
                db_size_gib: self.db_size_gib(),
                gamma0,
                gamma1,
                gamma2,
                config: Config {
                    n: DEFAULT_N,
                    base2k: DEFAULT_BASE2K,
                    k: DEFAULT_K,
                    collapse: Collapse::Recursion {
                        gamma0,
                        gamma1,
                        gamma2,
                    },
                    _phantom: PhantomData,
                },
                layout: DatabaseLayout::new(self.rows(), self.cols()),
            }),
        }
    }

    pub fn interpolation(self) -> Option<DefaultPirInterpolationParams32B> {
        match self.resolve() {
            DefaultPirConfig32B::Interpolation(params) => Some(params),
            DefaultPirConfig32B::Recursion(_) => None,
        }
    }

    pub fn recursion(self) -> Option<DefaultPirRecursionParams32B> {
        match self.resolve() {
            DefaultPirConfig32B::Recursion(params) => Some(params),
            DefaultPirConfig32B::Interpolation(_) => None,
        }
    }
}

/// The second-dimension *collapse* (the reduction over the `nb_matrices` panels),
/// carried as data so a single [`Parameters`] value can describe either PIR
/// construction. The shared cryptosystem (ring, base2k regimes, gadget shape) is
/// identical for both; only this enum differs, so switching constructions is a
/// one-field change. The database *shape* (`t`, `cols`, `nb_matrices`) is **not**
/// here — it lives in the database layout, the single source of truth for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collapse {
    /// InsPIRe: interpolate the panels into a degree-`t` polynomial and evaluate
    /// it at the client's GGSW root (`horner_evaluate`). No extra knobs — the
    /// gadget shape (`dnum`/`dsize`) and degree come from the shared regime / DB.
    Interpolation,
    /// InsPIRe²: recursively partial-pack the panels' packed results. `γ0` is the
    /// first-level record-packing parameter, `γ1`/`γ2` the second-level mask- and
    /// body-digit packing parameters (each a power of two `≤ n/2`).
    Recursion {
        gamma0: usize,
        gamma1: usize,
        gamma2: usize,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub struct Config<B, P>
where
    P: Payload<B>,
{
    pub(crate) n: usize,
    pub(crate) base2k: usize,
    pub(crate) k: usize,
    pub(crate) collapse: Collapse,
    pub(crate) _phantom: PhantomData<(B, P)>,
}

impl<B, P> Copy for Config<B, P> where P: Payload<B> {}

impl<B, P> Clone for Config<B, P>
where
    P: Payload<B>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, P> Config<B, P>
where
    P: Payload<B>,
{
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn collapse(&self) -> Collapse {
        self.collapse
    }

    pub fn column_height(&self) -> usize {
        match self.collapse {
            Collapse::Interpolation => self.n,
            Collapse::Recursion { gamma0, .. } => gamma0,
        }
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new<BE: Backend>(self) -> Parameters<BE, B, P>
    where
        Module<BE>: ModuleNew<BE>,
    {
        assert!(
            P::EXPONENT <= self.column_height(),
            "payload digits must fit within one scheme column"
        );
        if let Collapse::Recursion { gamma0, .. } = self.collapse {
            assert!(
                gamma0.is_multiple_of(P::EXPONENT),
                "Recursion gamma0 must be a whole number of payloads"
            );
        }
        let module = Module::<BE>::new(self.n as u64);
        Parameters {
            params: self,
            module,
        }
    }
}
