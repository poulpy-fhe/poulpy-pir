mod aggregate;

pub use crate::interpolation::interpolate_tmp_bytes;
pub use aggregate::AggregateLWE;
pub fn interpolate<BE, V>(
    module: &poulpy_hal::layouts::Module<BE>,
    y: &mut [V],
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) where
    BE: poulpy_hal::layouts::Backend,
    poulpy_hal::layouts::Module<BE>: poulpy_hal::api::ModuleN
        + poulpy_hal::api::VecZnxAddAssignBackend<BE>
        + poulpy_hal::api::VecZnxRotateBackend<BE>
        + poulpy_hal::api::VecZnxSubBackend<BE>,
    V: poulpy_hal::layouts::VecZnxToBackendMut<BE>
        + poulpy_hal::layouts::VecZnxToBackendRef<BE>
        + poulpy_hal::layouts::ZnxInfos,
{
    crate::interpolation::interpolate_columns(module, y, |v| v, |v| v, 0, scratch);
}
