import pytest
import yaml
import random
from environments.defend import ZoneDefense
from simulation.particles import Types


@pytest.fixture(params=["environments/config/defend_config.yaml"])
def env_config(request):

    with open(request.param) as f:
        return yaml.load(f, yaml.SafeLoader)


def test_init_gym(env_config):

    env = ZoneDefense(env_config)
    obs = env.reset()
    all_done = False

    while not all_done:

        enemies = env.simulator.get_all_of_type(Types.ENEMY)
        action = random.choice(list(range(len(enemies) + 1)))
        obs, rwd, done, info = env.step({"agent": action})
        all_done = done["__all__"]
    assert obs
