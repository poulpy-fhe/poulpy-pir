use poulpy_hal::layouts::{Backend, Module};

use crate::server::default::{
    InterpolationOfflineStepDefault, InterpolationOnlineStepDefault, InterpolationSetupStepDefault,
    RecursionOfflineStepDefault, RecursionOnlineStepDefault, RecursionSetupStepDefault,
};

/// # Safety
/// Sealed marker: an implementor asserts its [`InterpolationSetupStepDefault`]
/// impl upholds the documented setup-step semantics for the backend's layouts.
pub(crate) unsafe trait InterpolationSetupStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> InterpolationSetupStepImpl<BE> for BE where
    Module<BE>: InterpolationSetupStepDefault<BE>
{
}

/// # Safety
/// Sealed marker: an implementor asserts its [`InterpolationOfflineStepDefault`]
/// impl upholds the documented offline-step semantics for the backend's layouts.
pub(crate) unsafe trait InterpolationOfflineStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> InterpolationOfflineStepImpl<BE> for BE where
    Module<BE>: InterpolationOfflineStepDefault<BE>
{
}

/// # Safety
/// Sealed marker: an implementor asserts its [`InterpolationOnlineStepDefault`]
/// impl upholds the documented online-step semantics for the backend's layouts.
pub(crate) unsafe trait InterpolationOnlineStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> InterpolationOnlineStepImpl<BE> for BE where
    Module<BE>: InterpolationOnlineStepDefault<BE>
{
}

/// # Safety
/// Sealed marker: an implementor asserts its [`RecursionSetupStepDefault`]
/// impl upholds the documented setup-step semantics for the backend's layouts.
pub(crate) unsafe trait RecursionSetupStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> RecursionSetupStepImpl<BE> for BE where
    Module<BE>: RecursionSetupStepDefault<BE>
{
}

/// # Safety
/// Sealed marker: an implementor asserts its [`RecursionOfflineStepDefault`]
/// impl upholds the documented offline-step semantics for the backend's layouts.
pub(crate) unsafe trait RecursionOfflineStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> RecursionOfflineStepImpl<BE> for BE where
    Module<BE>: RecursionOfflineStepDefault<BE>
{
}

/// # Safety
/// Sealed marker: an implementor asserts its [`RecursionOnlineStepDefault`]
/// impl upholds the documented online-step semantics for the backend's layouts.
pub(crate) unsafe trait RecursionOnlineStepImpl<BE: Backend>: Backend {}

unsafe impl<BE: Backend> RecursionOnlineStepImpl<BE> for BE where
    Module<BE>: RecursionOnlineStepDefault<BE>
{
}
