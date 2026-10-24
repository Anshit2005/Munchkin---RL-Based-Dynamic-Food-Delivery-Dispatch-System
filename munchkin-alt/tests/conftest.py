import pytest

from baselines import MinETAPolicy, MostUrgentOrderPolicy, NearestDriverPolicy
from environment.config import load_scenarios
from simulation.policy import NeverAssignPolicy, RandomValidPolicy


@pytest.fixture(scope="session")
def scenarios():
    return load_scenarios()


@pytest.fixture
def all_policies():
    return [NearestDriverPolicy(), MostUrgentOrderPolicy(), MinETAPolicy(), RandomValidPolicy(0),
            NeverAssignPolicy()]
