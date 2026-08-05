use std::marker::PhantomData;

use poulpy_cpu_ref::FFT64Ref;

use crate::{
    client::Response,
    config::{Collapse, Config, DEFAULT_BASE2K, DEFAULT_K},
    database::DatabaseLayout,
    payload::{U256P65535, U256P65536},
    server::Query,
};

type BE = FFT64Ref;

#[test]
fn query_read_rejects_interpolation_length_mismatch_before_allocation() {
    let config = Config::<U256P65535> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<U256P65535>::new(64, 64);
    let params = config.new::<BE>();
    let mut bytes = Vec::new();
    bytes.push(0); // TAG_INTERPOLATION
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let Err(err) = Query::read_from(&mut &bytes[..], &params, layout) else {
        panic!("malicious interpolation query length was accepted");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn query_read_rejects_recursion_length_mismatch_before_allocation() {
    let config = Config::<U256P65536> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0: 32,
            gamma1: 32,
            gamma2: 16,
        },
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<U256P65536>::new(64, 64);
    let params = config.new::<BE>();
    let mut bytes = Vec::new();
    bytes.push(1); // TAG_RECURSION
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let Err(err) = Query::read_from(&mut &bytes[..], &params, layout) else {
        panic!("malicious recursion query length was accepted");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn response_read_rejects_recursion_length_mismatch_before_allocation() {
    let config = Config::<U256P65536> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0: 32,
            gamma1: 32,
            gamma2: 16,
        },
        _phantom: PhantomData,
    };
    let params = config.new::<BE>();
    let mut bytes = Vec::new();
    bytes.push(1); // TAG_RECURSION
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let Err(err) = Response::read_from(&mut &bytes[..], &params) else {
        panic!("malicious recursion response length was accepted");
    };
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
