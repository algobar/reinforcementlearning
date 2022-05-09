import pytest
import numpy
from simulation.behaviors import (
    straight_line_path_2d,
    plan_2d_intercept,
)

from simulation.particles import create_position


def test_straight_line_path():

    current = create_position(0, 0, 0)
    goal = create_position(5, 0, 0)
    speed = 1
    delta_t = 0.5

    straight_line_path_2d(current=current, end=goal, speed=speed, timestep=delta_t)

    assert current[0] == 0.5

    ret = False
    for i in range(10):
        ret = straight_line_path_2d(
            current=current, end=goal, speed=speed, timestep=delta_t
        )

    assert current[0] == 5
    assert current[1] == 0
    assert current[2] == 0
    assert ret


def test_intercept_2d():

    tgt_pos = create_position(4, 1, 0)
    tgt_dest = create_position(6, 7, 0)
    int_pos = create_position(1, 5, 0)
    tgt_speed = 1.0
    int_speed = 1.1
    result = plan_2d_intercept(
        interceptor_pos=int_pos,
        target_pos=tgt_pos,
        target_dest=tgt_dest,
        interceptor_speed=int_speed,
        target_speed=tgt_speed,
    )

    assert result[0] is not None


def test_collision():

    tgt_pos = create_position(0, 0, 0)
    tgt_goal = create_position(10, 10, 0)
    tgt_speed = 1

    interceptor = create_position(10, 0, 0)
    interceptor_speed = 2

    result = plan_2d_intercept(
        interceptor, tgt_pos, tgt_goal, interceptor_speed, tgt_speed
    )

    assert result[0] is not None

    int_goal = result[0]
    delta_t = 0.01
    # run a fake simulation
    min_distance = 10
    for i in range(1000):

        straight_line_path_2d(tgt_pos, tgt_goal, tgt_speed, delta_t)
        straight_line_path_2d(interceptor, int_goal, interceptor_speed, delta_t)

        dist_between = numpy.linalg.norm(interceptor - tgt_pos)

        if dist_between < min_distance:
            min_distance = dist_between

    print("recorded closest approach", min_distance)
    assert min_distance < 0.05
