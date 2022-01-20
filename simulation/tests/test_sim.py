from simulation.particles import SimpleWorld, create_position, Particle
from simulation.behaviors import go_to_point_2d, magnitude


def test_simulation():

    sim = SimpleWorld(time=0)

    pos = create_position(0, 0, 0)
    end = create_position(10, 10, 0)

    obj_a = Particle(name="test", position=pos, speed=1)
    obj_a.add_behavior(go_to_point_2d(end, obj_a.speed))

    assert obj_a.tasked

    sim.add_object(obj_a.name, obj_a)

    t = 0.1

    for i in range(200):

        sim.update(timestep=t, actions={})

    assert not obj_a.tasked
