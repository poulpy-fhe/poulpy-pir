use poulpy_hal::layouts::Backend;

use crate::server::default::{
    InterpolationOfflineStepDefault, InterpolationOnlineStepDefault, InterpolationSetupStepDefault,
    RecursionOfflineStepDefault, RecursionOnlineStepDefault, RecursionSetupStepDefault,
};

pub(crate) trait InterpolationSetupStep<BE: Backend>:
    InterpolationSetupStepDefault<BE>
{
}

pub(crate) trait InterpolationOfflineStep<BE: Backend>:
    InterpolationOfflineStepDefault<BE>
{
}

pub(crate) trait InterpolationOnlineStep<BE: Backend>:
    InterpolationOnlineStepDefault<BE>
{
}

pub(crate) trait InterpolationServerModule<BE: Backend>:
    InterpolationSetupStep<BE> + InterpolationOfflineStep<BE> + InterpolationOnlineStep<BE>
{
}

pub(crate) trait RecursionSetupStep<BE: Backend>: RecursionSetupStepDefault<BE> {}

pub(crate) trait RecursionOfflineStep<BE: Backend>: RecursionOfflineStepDefault<BE> {}

pub(crate) trait RecursionOnlineStep<BE: Backend>: RecursionOnlineStepDefault<BE> {}

pub(crate) trait RecursionServerModule<BE: Backend>:
    RecursionSetupStep<BE> + RecursionOfflineStep<BE> + RecursionOnlineStep<BE>
{
}
