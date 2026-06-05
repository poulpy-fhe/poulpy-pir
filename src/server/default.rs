use poulpy_core::{
    GLWEDecrypt, GLWEExpandLWEMatrix, GLWEMaskFill, GLWEMaskFillDefault, GLWENormalize,
    layouts::{
        GGLWEPreparedFactory, GGSWPreparedFactory, GLWEDecompress, GLWESecretPreparedFactory,
        ModuleCoreAlloc, ModuleCoreCompressedAlloc, SecretConversion,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScalarZnxAutomorphismBackend, VecZnxAddAssignBackend, VecZnxAlloc,
        VecZnxAutomorphismBackend, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes, VecZnxRshAssignBackend,
        VecZnxRshTmpBytes, VecZnxTransposeBackend, VecZnxZeroBackend, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPrepare, VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{Backend, GaloisElement},
};

use crate::{
    interpolation::{HornerEvaluation, MonomialInterpolation},
    packing::{Packing, PackingKeysGenerate, PackingMaskAggregation},
};

pub(crate) trait InterpolationSetupStepDefault<BE: Backend>:
    ModuleNew<BE>
    + ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + GLWEExpandLWEMatrix<BE>
    + GLWEMaskFill<BE>
    + VecZnxNormalize<BE>
    + VecZnxNormalizeTmpBytes
    + VecZnxZeroBackend<BE>
{
}

impl<BE, M> InterpolationSetupStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleNew<BE>
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + GLWEExpandLWEMatrix<BE>
        + GLWEMaskFill<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxZeroBackend<BE>,
{
}

pub(crate) trait InterpolationOfflineStepDefault<BE: Backend>:
    ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + MonomialInterpolation<BE>
    + PackingMaskAggregation<BE>
    + Packing<BE>
    + PackingKeysGenerate<BE>
    + GGSWPreparedFactory<BE>
    + GLWEExpandLWEMatrix<BE>
    + GLWEMaskFill<BE>
    + VecZnxNormalize<BE>
    + VecZnxNormalizeTmpBytes
    + VecZnxAddAssignBackend<BE>
    + VecZnxCopyBackend<BE>
    + VecZnxZeroBackend<BE>
    + VmpPrepare<BE>
    + VmpPrepareTmpBytes
    + VmpZero<BE>
{
}

impl<BE, M> InterpolationOfflineStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + MonomialInterpolation<BE>
        + PackingMaskAggregation<BE>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + GGSWPreparedFactory<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEMaskFill<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
}

pub(crate) trait InterpolationOnlineStepDefault<BE: Backend>:
    ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + HornerEvaluation<BE>
    + Packing<BE>
    + PackingKeysGenerate<BE>
    + GGSWPreparedFactory<BE>
    + VecZnxNormalize<BE>
    + VecZnxAddAssignBackend<BE>
    + VecZnxCopyBackend<BE>
    + VecZnxZeroBackend<BE>
    + VmpPrepare<BE>
    + VmpPrepareTmpBytes
    + VmpZero<BE>
{
}

impl<BE, M> InterpolationOnlineStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + HornerEvaluation<BE>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxNormalize<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
}

pub(crate) trait RecursionSetupStepDefault<BE: Backend>:
    ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + ModuleCoreCompressedAlloc
    + SecretConversion<BE>
    + GLWESecretPreparedFactory<BE>
    + ScalarZnxAutomorphismBackend<BE>
    + PackingKeysGenerate<BE>
    + GaloisElement
    + GGLWEPreparedFactory<BE>
    + GLWEMaskFillDefault<BE>
    + VecZnxAlloc<BE>
    + VecZnxAutomorphismBackend<BE>
    + VecZnxBigBytesOf
    + VecZnxBigNormalize<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxDftAddAssign<BE>
    + VecZnxDftAlloc<BE>
    + VecZnxDftApply<BE>
    + VecZnxDftAutomorphism<BE>
    + VecZnxDftBytesOf
    + VecZnxDftZero<BE>
    + VecZnxIdftApply<BE>
    + VecZnxIdftApplyTmpBytes
    + VecZnxRotateAssignBackend<BE>
    + VecZnxRotateAssignTmpBytes
    + VecZnxRshAssignBackend<BE>
    + VecZnxRshTmpBytes
    + VecZnxTransposeBackend<BE>
    + VmpApplyDftToDft<BE>
    + VmpApplyDftToDftTmpBytes
{
}

impl<BE, M> RecursionSetupStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + ModuleCoreCompressedAlloc
        + SecretConversion<BE>
        + GLWESecretPreparedFactory<BE>
        + ScalarZnxAutomorphismBackend<BE>
        + PackingKeysGenerate<BE>
        + GaloisElement
        + GGLWEPreparedFactory<BE>
        + GLWEMaskFillDefault<BE>
        + VecZnxAlloc<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRotateAssignTmpBytes
        + VecZnxRshAssignBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxTransposeBackend<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
{
}

pub(crate) trait RecursionOfflineStepDefault<BE: Backend>:
    ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + Packing<BE>
    + PackingKeysGenerate<BE>
    + PackingMaskAggregation<BE>
    + GLWEExpandLWEMatrix<BE>
    + GLWEMaskFill<BE>
    + GLWENormalize<BE>
    + GLWEDecrypt<BE>
    + VecZnxNormalize<BE>
    + VecZnxNormalizeTmpBytes
    + VecZnxAddAssignBackend<BE>
    + VecZnxCopyBackend<BE>
    + VecZnxZeroBackend<BE>
    + VmpPrepare<BE>
    + VmpPrepareTmpBytes
    + VmpZero<BE>
{
}

impl<BE, M> RecursionOfflineStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + PackingMaskAggregation<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEMaskFill<BE>
        + GLWENormalize<BE>
        + GLWEDecrypt<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
}

pub(crate) trait RecursionOnlineStepDefault<BE: Backend>:
    ModuleN
    + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
    + Packing<BE>
    + PackingKeysGenerate<BE>
    + PackingMaskAggregation<BE>
    + GLWEExpandLWEMatrix<BE>
    + GLWEDecompress<Backend = BE>
    + GLWENormalize<BE>
    + VecZnxNormalize<BE>
    + VecZnxNormalizeTmpBytes
    + VecZnxAddAssignBackend<BE>
    + VecZnxCopyBackend<BE>
    + VecZnxZeroBackend<BE>
    + VmpPrepare<BE>
    + VmpPrepareTmpBytes
    + VmpZero<BE>
{
}

impl<BE, M> RecursionOnlineStepDefault<BE> for M
where
    BE: Backend,
    M: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + PackingMaskAggregation<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEDecompress<Backend = BE>
        + GLWENormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
}
