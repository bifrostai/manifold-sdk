import numpy as np

from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.recipes.function import FunctionEndpoint


def test_function_endpoint_returns_the_prediction() -> None:
    action = Action.from_array(np.zeros(7, dtype=np.float32))

    def predict(_observation: Observation) -> Action:
        return action

    signature = PolicySignature(
        action_space=FRANKA_EE_DELTA.action,
        proprioception=FRANKA_EE_DELTA.proprioception,
    )
    endpoint = FunctionEndpoint(predict, signature)
    assert endpoint.session().infer(Observation()) is action
