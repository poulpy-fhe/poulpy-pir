use poulpy_hal::layouts::{Backend, Module};

use crate::server::{
    api::{
        InterpolationOfflineStep, InterpolationOnlineStep, InterpolationServerModule,
        InterpolationSetupStep, RecursionOfflineStep, RecursionOnlineStep, RecursionServerModule,
        RecursionSetupStep,
    },
    default::{
        InterpolationOfflineStepDefault, InterpolationOnlineStepDefault,
        InterpolationSetupStepDefault, RecursionOfflineStepDefault, RecursionOnlineStepDefault,
        RecursionSetupStepDefault,
    },
    oep::{
        InterpolationOfflineStepImpl, InterpolationOnlineStepImpl, InterpolationSetupStepImpl,
        RecursionOfflineStepImpl, RecursionOnlineStepImpl, RecursionSetupStepImpl,
    },
};

impl<BE> InterpolationSetupStep<BE> for Module<BE>
where
    BE: Backend + InterpolationSetupStepImpl<BE>,
    Module<BE>: InterpolationSetupStepDefault<BE>,
{
}

impl<BE> InterpolationOfflineStep<BE> for Module<BE>
where
    BE: Backend + InterpolationOfflineStepImpl<BE>,
    Module<BE>: InterpolationOfflineStepDefault<BE>,
{
}

impl<BE> InterpolationOnlineStep<BE> for Module<BE>
where
    BE: Backend + InterpolationOnlineStepImpl<BE>,
    Module<BE>: InterpolationOnlineStepDefault<BE>,
{
}

impl<BE> InterpolationServerModule<BE> for Module<BE>
where
    BE: Backend,
    Module<BE>:
        InterpolationSetupStep<BE> + InterpolationOfflineStep<BE> + InterpolationOnlineStep<BE>,
{
}

impl<BE> RecursionSetupStep<BE> for Module<BE>
where
    BE: Backend + RecursionSetupStepImpl<BE>,
    Module<BE>: RecursionSetupStepDefault<BE>,
{
}

impl<BE> RecursionOfflineStep<BE> for Module<BE>
where
    BE: Backend + RecursionOfflineStepImpl<BE>,
    Module<BE>: RecursionOfflineStepDefault<BE>,
{
}

impl<BE> RecursionOnlineStep<BE> for Module<BE>
where
    BE: Backend + RecursionOnlineStepImpl<BE>,
    Module<BE>: RecursionOnlineStepDefault<BE>,
{
}

impl<BE> RecursionServerModule<BE> for Module<BE>
where
    BE: Backend,
    Module<BE>: RecursionSetupStep<BE> + RecursionOfflineStep<BE> + RecursionOnlineStep<BE>,
{
}
