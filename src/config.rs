use std::marker::PhantomData;

use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module},
};

use crate::{
    parameters::Parameters,
    payload::{P65535, P65536, Payload},
};

pub static INSPIRE_INT_32B: Config<[u8; 32], P65535<[u8; 32]>> = Config {
    n: 2048,
    base2k: 18,
    k: 54,
    collapse: Collapse::Interpolation,
    _phantom: PhantomData,
};

pub static INSPIRE_REC_32B: Config<[u8; 32], P65536<[u8; 32]>> = Config {
    n: 2048,
    base2k: 18,
    k: 54,
    collapse: Collapse::Recursion {
        gamma0: 32,
        gamma1: 1024,
        gamma2: 32,
    },
    _phantom: PhantomData,
};

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
