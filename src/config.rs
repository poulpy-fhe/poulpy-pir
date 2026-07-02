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

/// Collapse family for a default parameterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultScheme {
    /// InsPIRe (interpolation collapse), payload `U256P65535`.
    Interpolation,
    /// InsPIRe² (recursion collapse), payload `U256P65536`, with the given `γ0`
    /// (fixed `γ1=1024`, `γ2=γ0`). `γ0 ∈ {16, 32, 64}`.
    Recursion { gamma0: usize },
}

/// A ready-made 32-byte PIR parameterization: a [`DefaultScheme`], a power-of-two
/// database size in GiB, and the first database dimension `cols` (the layout
/// trade-off point). The second dimension is derived: `rows = 2^29 · db_gib /
/// cols`, so `rows · cols` is exactly the DB's `u16`-coefficient count.
///
/// [`Self::all`] enumerates the full grid — every `cols` in the per-size window
/// ([`Self::cols_window`], anchored on the paper's Table 2 / Appendix Table 8 and
/// log2-interpolated for the intermediate sizes) crossed with all four schemes.
/// Use [`Self::resolve`] when the construction is selected dynamically, or
/// [`Self::interpolation`] / [`Self::recursion`] when the caller knows the payload
/// type. [`Self::canonical`] picks the single `rows = 2^16` shape per size (the
/// historical uniform default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultPirParameters32B {
    pub scheme: DefaultScheme,
    pub db_gib: usize,
    pub cols: usize,
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
    /// The four default schemes (InsPIRe + InsPIRe² for `γ0 ∈ {16, 32, 64}`).
    pub const SCHEMES: [DefaultScheme; 4] = [
        DefaultScheme::Interpolation,
        DefaultScheme::Recursion { gamma0: 16 },
        DefaultScheme::Recursion { gamma0: 32 },
        DefaultScheme::Recursion { gamma0: 64 },
    ];

    /// The power-of-two database sizes (GiB) covered by the defaults.
    pub const DB_SIZES_GIB: [usize; 6] = [1, 2, 4, 8, 16, 32];

    /// Every default parameterization (108): the 27 layout shapes per scheme
    /// crossed with the four schemes, ordered scheme-major, then by DB size, then
    /// by ascending `cols`. Hard-coded (not generated) so the whole set is
    /// auditable at a glance; `default_32b_grid_is_valid_and_covers_table2` checks
    /// every entry against [`Self::cols_window`] and the paper's Table 2.
    pub const ALL: [Self; 108] = [
        // InsPIRe (interpolation)
        Self { scheme: DefaultScheme::Interpolation, db_gib: 1, cols: 4096 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 1, cols: 8192 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 1, cols: 16384 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 1, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 2, cols: 4096 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 2, cols: 8192 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 2, cols: 16384 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 2, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 2, cols: 65536 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 4, cols: 8192 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 4, cols: 16384 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 4, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 4, cols: 65536 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 8, cols: 8192 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 8, cols: 16384 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 8, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 8, cols: 65536 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 8, cols: 131072 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 16, cols: 16384 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 16, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 16, cols: 65536 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 16, cols: 131072 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 16, cols: 262144 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 32, cols: 32768 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 32, cols: 65536 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 32, cols: 131072 },
        Self { scheme: DefaultScheme::Interpolation, db_gib: 32, cols: 262144 },
        // InsPIRe² γ0=16
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 1, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 1, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 1, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 1, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 2, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 2, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 2, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 2, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 2, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 4, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 4, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 4, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 4, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 8, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 8, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 8, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 8, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 8, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 16, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 16, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 16, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 16, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 16, cols: 262144 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 32, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 32, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 32, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 16 }, db_gib: 32, cols: 262144 },
        // InsPIRe² γ0=32
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 1, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 1, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 1, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 1, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 2, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 2, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 2, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 2, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 2, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 4, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 4, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 4, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 4, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 8, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 8, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 8, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 8, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 8, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 16, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 16, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 16, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 16, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 16, cols: 262144 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 32, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 32, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 32, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 32 }, db_gib: 32, cols: 262144 },
        // InsPIRe² γ0=64
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 1, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 1, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 1, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 1, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 2, cols: 4096 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 2, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 2, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 2, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 2, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 4, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 4, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 4, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 4, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 8, cols: 8192 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 8, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 8, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 8, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 8, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 16, cols: 16384 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 16, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 16, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 16, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 16, cols: 262144 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 32, cols: 32768 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 32, cols: 65536 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 32, cols: 131072 },
        Self { scheme: DefaultScheme::Recursion { gamma0: 64 }, db_gib: 32, cols: 262144 },
    ];

    /// Convenience `Vec` view of [`Self::ALL`].
    pub fn all() -> Vec<Self> {
        Self::ALL.to_vec()
    }

    /// The single canonical shape for a size: `rows = 2^16`, i.e.
    /// `cols = 2^13 · db_gib` (the historical uniform default; always a member of
    /// [`Self::all`]).
    pub const fn canonical(scheme: DefaultScheme, db_gib: usize) -> Self {
        Self {
            scheme,
            db_gib,
            cols: (1 << 13) * db_gib,
        }
    }

    /// Inclusive `log2(cols)` window for a database size. Anchored on the paper's
    /// Table 2 (`1/8/32 GiB → [12,15] / [13,17] / [15,18]`) and log2-interpolated
    /// (rounded to a power of two) for the extrapolated `2/4/16 GiB` sizes. Every
    /// power-of-two `cols` in the window is a valid layout (`rows = total/cols`
    /// stays a multiple of `n` and of every `γ0`).
    pub const fn cols_window(db_gib: usize) -> (u32, u32) {
        match db_gib {
            1 => (12, 15),
            2 => (12, 16),
            4 => (13, 16),
            8 => (13, 17),
            16 => (14, 18),
            32 => (15, 18),
            _ => panic!("unsupported database size (expected a power of two in 1..=32 GiB)"),
        }
    }

    /// Total `u16` coefficients in the database (`2^29 · db_gib`).
    pub const fn total_u16(self) -> usize {
        (1usize << 29) * self.db_gib
    }

    pub const fn db_size_gib(self) -> usize {
        self.db_gib
    }

    /// First database dimension (the one-hot width).
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Second database dimension, derived so `rows · cols` fills the DB exactly.
    pub const fn rows(self) -> usize {
        self.total_u16() / self.cols
    }

    /// `(rows, cols)` database layout.
    pub const fn layout_rows_cols(self) -> (usize, usize) {
        (self.rows(), self.cols)
    }

    pub const fn collapse(self) -> Collapse {
        match self.scheme {
            DefaultScheme::Interpolation => Collapse::Interpolation,
            DefaultScheme::Recursion { gamma0 } => Collapse::Recursion {
                gamma0,
                gamma1: 1024,
                gamma2: gamma0,
            },
        }
    }

    pub const fn gamma0(self) -> Option<usize> {
        match self.scheme {
            DefaultScheme::Interpolation => None,
            DefaultScheme::Recursion { gamma0 } => Some(gamma0),
        }
    }

    pub const fn gamma1(self) -> Option<usize> {
        match self.scheme {
            DefaultScheme::Interpolation => None,
            DefaultScheme::Recursion { .. } => Some(1024),
        }
    }

    pub const fn gamma2(self) -> Option<usize> {
        self.gamma0()
    }

    /// A stable, parseable identifier, e.g. `"InsPIRe-1GiB-c8192"` or
    /// `"InsPIRe2-g64-8GiB-c16384"`. Round-trips through [`Self::from_name`].
    pub fn name(self) -> String {
        match self.scheme {
            DefaultScheme::Interpolation => format!("InsPIRe-{}GiB-c{}", self.db_gib, self.cols),
            DefaultScheme::Recursion { gamma0 } => {
                format!("InsPIRe2-g{gamma0}-{}GiB-c{}", self.db_gib, self.cols)
            }
        }
    }

    /// Parse a [`Self::name`]; `None` if it isn't a member of [`Self::all`].
    pub fn from_name(name: &str) -> Option<Self> {
        Self::all().into_iter().find(|v| v.name() == name)
    }

    /// Actual serialized **query** size in bytes for this variant, computed from
    /// the cryptosystem parameters + layout via [`Config::query_size`] — the
    /// single source of truth (no hand-fitted per-DB numbers). Backend-free.
    pub fn query_bytes(self) -> usize {
        match self.resolve() {
            DefaultPirConfig32B::Interpolation(p) => p.config.query_size(p.layout).total_size(),
            DefaultPirConfig32B::Recursion(p) => p.config.query_size(p.layout).total_size(),
        }
    }

    /// Actual serialized **response** size in bytes for this variant, via
    /// [`Config::response_size`]. Backend-free.
    pub fn response_bytes(self) -> usize {
        match self.resolve() {
            DefaultPirConfig32B::Interpolation(p) => p.config.response_size(p.layout).total_size(),
            DefaultPirConfig32B::Recursion(p) => p.config.response_size(p.layout).total_size(),
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
