use std::marker::PhantomData;

use poulpy_cpu_ref::FFT64Ref;

use crate::{
    client::{Client, RecursionResponse, Response},
    config::{Collapse, Config, DEFAULT_BASE2K, DEFAULT_K},
    database::DatabaseLayout,
    error::PirError,
    payload::{U256P65535, U256P65536},
    server::Server,
};

type BE = FFT64Ref;

#[test]
fn fallible_layout_rejects_zero_dimensions() {
    let Err(err) = DatabaseLayout::<U256P65535>::try_new(0, 64) else {
        panic!("zero-row layout was accepted");
    };
    assert!(matches!(err, PirError::InvalidLayout { .. }));
}

#[test]
fn fallible_config_rejects_payload_that_does_not_fit_gamma0() {
    let config = Config::<U256P65536> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Recursion {
            gamma0: 40,
            gamma1: 32,
            gamma2: 16,
        },
        _phantom: PhantomData,
    };
    let Err(err) = config.try_new::<BE>() else {
        panic!("invalid gamma0 config was accepted");
    };
    assert!(matches!(err, PirError::InvalidConfig { .. }));
}

#[test]
fn fallible_query_rejects_out_of_range_payload_index() {
    let config = Config::<U256P65535> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<U256P65535>::new(64, 64);
    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let Err(err) = client.try_query(layout.num_payloads(config.n())) else {
        panic!("out-of-range query was accepted");
    };
    assert!(matches!(err, PirError::PayloadOutOfBounds { .. }));
}

#[test]
fn fallible_shard_update_rejects_out_of_range_write() {
    let config = Config::<U256P65535> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<U256P65535>::new(64, 64);
    let mut server = Server::<BE, U256P65535>::new(config, layout);
    let Err(err) = server.try_update_shard(layout.num_payloads(config.n()), &[[0u8; 32]]) else {
        panic!("out-of-range shard update was accepted");
    };
    assert!(matches!(err, PirError::ShardOutOfBounds { .. }));
}

#[test]
fn fallible_server_rejects_query_for_other_collapse() {
    let interp_config = Config::<U256P65535> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let rec_config = Config::<U256P65536> {
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
    let interp_layout = DatabaseLayout::<U256P65535>::new(64, 64);
    let rec_layout = DatabaseLayout::<U256P65536>::new(64, 64);
    let mut server = Server::<BE, U256P65535>::new(interp_config, interp_layout);
    let mut client = Client::<BE, U256P65536>::new(rec_config, rec_layout);
    let (query, _) = client.query(0);

    let Err(err) = server.try_respond(&query) else {
        panic!("mismatched query variant was accepted");
    };
    assert!(matches!(err, PirError::WrongQueryVariant { .. }));
}

#[test]
fn fallible_client_rejects_response_for_other_collapse() {
    let config = Config::<U256P65535> {
        n: 64,
        base2k: DEFAULT_BASE2K,
        k: DEFAULT_K,
        collapse: Collapse::Interpolation,
        _phantom: PhantomData,
    };
    let layout = DatabaseLayout::<U256P65535>::new(64, 64);
    let mut client = Client::<BE, U256P65535>::new(config, layout);
    let (_, state) = client.query(0);
    let response = Response::Recursion(RecursionResponse::new(Vec::new(), Vec::new()));

    let Err(err) = client.try_decode(&response, &state) else {
        panic!("mismatched response variant was accepted");
    };
    assert!(matches!(err, PirError::WrongResponseVariant { .. }));
}
