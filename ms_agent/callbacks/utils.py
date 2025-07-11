# Copyright (c) Alibaba, Inc. and its affiliates.
from ms_agent.callbacks.artifact_callback import ArtifactCallback
from ms_agent.callbacks.input_callback import InputCallback

callbacks_mapping = {
    'input_callback': InputCallback,
    'artifact_callback': ArtifactCallback,
}
