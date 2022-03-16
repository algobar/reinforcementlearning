import pytest
import yaml
import random
from environments.gyms.defend import ZoneDefense
from simulation.particles import Types


@pytest.fixture(params=["config/defend_env.yaml"])
def env_config(request):
    """Returns the environment config"""
    with open(request.param) as f:
        return yaml.load(f, yaml.SafeLoader)


def test_init_gym(env_config):
    """Test that we can initialize the gym correctly"""

    env = ZoneDefense(env_config)
    obs = env.reset()
    all_done = False

    while not all_done:

        enemies = env.simulator.get_all_of_type(Types.ENEMY)
        get_action = lambda: random.choice(list(range(len(enemies) + 1)))
        actions = {a: get_action() for a in obs}
        obs, rwd, done, info = env.step(actions)
        all_done = done["__all__"]
    assert obs
