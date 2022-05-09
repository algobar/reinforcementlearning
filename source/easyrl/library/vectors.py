import math
import numpy
from typing import Tuple

from easyrl.simulation.particles import Particle

ABSOLUTE_NORTH: numpy.array = numpy.array([0, 1, 0], dtype=numpy.float32)
Degrees = float


def copy_array(arr: numpy.array) -> numpy.array:
    """Return a copy of the array"""
    return numpy.copy(arr)


def magnitude(vec: numpy.array) -> float:
    """Find the magnitude of the vector

    Args:
        vec (numpy.array): [description]

    Returns:
        float: scalar representing magnitude
    """

    return numpy.linalg.norm(vec, axis=-1)


def get_unit_vector(array: numpy.array) -> numpy.array:
    """Return the unit vector of the array"""
    return array / magnitude(array)


def calculate_absolute_bearing(
    origin: numpy.array, destination: numpy.array
) -> float:
    """Calculate the bearing from the origin to the destination. Note that
    with the particle designation, this does not include any heading specific
    information, assuming that the bearing angle being calculated from the
    y axis.

    Args:
        origin (numpy.array): [description]
        destination (numpy.array): [description]

    Returns:
        float: angle in radians
    """

    delta = destination - origin
    unit_vec = delta / magnitude(delta)

    dot_prod = numpy.dot(unit_vec, ABSOLUTE_NORTH)

    return numpy.arccos(dot_prod)


def determinant_2d(a: float, b: float, c: float, d: float):
    """Calculates the determinant of a 2D matrix

    Args:
        a (float): [description]
        b (float): [description]
        c (float): [description]
        d (float): [description]

    Returns:
        [type]: [description]
    """
    return numpy.linalg.det([[a, b], [c, d]])


def calculate_2d_intercept(
    interceptor_pos: numpy.array,
    target_pos: numpy.array,
    target_dest: numpy.array,
    interceptor_speed: float,
    target_speed: float,
) -> Tuple[numpy.array, float]:
    """

    Args:
        interceptor_pos (numpy.array): [description]
        target_pos (numpy.array): [description]
        interceptor_speed (float): [description]
        target_speed (float): [description]

    Returns:
        Tuple: [description]

    Source:
    http://zulko.github.io/blog/2013/11/11/interception-of-a-linear-trajectory-with-constant-speed/

    Steps:

    triangle property:

    A --------------- B ----------- T
    \ a            b  /
     \               /
      \             /
       \           /
        \         /
         \       /
          \     /
           \ c /
            \ /
             C

    where A is the target's position
    B is the intercept point
    C is the interceptor's position
    T is the target's destination


    sin(a)/BC = sin(b)/AC = sin(c)/AB

    sin(a) = det(AC, AT)/(AC * AT)

    sin(c) = AB * sin(a) / BC =
        v_a * t * sin(a) / (v_c * t) = v_a * sin(a) / v_c

    where v_a is velocity of the target, and v_c is velocity of the interceptor

    finally,

    sin(b) = sin(pi - a - c) = sin(a + c)
    sin(b) = sin(a)cos(c) + cos(a)sin(c)
    sin(b) = sin(a)sqrt(1 - sin^2(c)) + sin(c)sqrt(1 - sin^2(a))

    """

    AC: numpy.array = interceptor_pos - target_pos
    AT: numpy.array = target_dest - target_pos

    determinant: float = determinant_2d(AC[0], AC[1], AT[0], AT[1])

    sin_a: float = determinant / (magnitude(AC) * magnitude(AT))
    sin_c: float = target_speed * sin_a / interceptor_speed

    if sin_c > 1:
        return None, None
    elif sin_a < 0.001:
        """
        case where the interceptor lies
        on same line as target's path,
        causing divide by zero error. Problem
        reduces to where they should meet along
        that line.
        """
        dist_between: float = magnitude(AC)
        time: float = dist_between / (interceptor_speed + target_speed)
        mag_ab: float = time * target_speed

    else:
        sin_b: float = sin_a * math.sqrt(1 - sin_c**2) + sin_c * math.sqrt(
            1 - sin_a**2
        )
        mag_ab: float = magnitude(AC) * sin_c / sin_b
        time: float = mag_ab / target_speed

    angle_between: float = math.atan2(AT[1], AT[0])

    delta_x: float = mag_ab * math.cos(angle_between)
    delta_y: float = mag_ab * math.sin(angle_between)

    return (target_pos + [delta_x, delta_y, 0], time)


def straight_line_path_2d(
    current: numpy.array, end: numpy.array, speed: float, timestep: float
) -> numpy.array:
    """Calculates the updated position based on the desired end position

    Args:
        current (Position): [description]
        end (Position): [description]
        speed (float): [description]
        timestep (float): [description]

    Returns:
        bool: True when reached within 0.01 radius of position, else False
    """
    delta_pos: numpy.array = end - current

    hypotenuse: float = magnitude(delta_pos)

    angle: float = math.atan2(delta_pos[1], delta_pos[0])

    step_size: float = speed * timestep

    if step_size > hypotenuse:
        step_size = hypotenuse

    delta_x: float = step_size * math.cos(angle)
    delta_y: float = step_size * math.sin(angle)

    delta_arr = numpy.array([delta_x, delta_y, 0], dtype=current.dtype)
    current += delta_arr

    return current


def calculate_aspect(
    originator_heading: numpy.array,
    originator_position: numpy.array,
    target_position: numpy.array,
) -> Degrees:
    """Calculate aspect"""

    diff_pos = target_position - originator_position
    reverse_heading = originator_heading * -1

    angle_cosine = numpy.dot(diff_pos, reverse_heading) / (
        magnitude(diff_pos) * magnitude(reverse_heading)
    )

    if angle_cosine > 1.0:
        angle_cosine = 1.0
    angle_rad = numpy.arccos(angle_cosine)
    angle_deg = numpy.rad2deg(angle_rad)
    direction = -1 if numpy.cross(reverse_heading, diff_pos)[0] < 0 else 1

    return angle_deg * direction


def calculate_aspect_offset(
    direction: numpy.array, offset_deg: float
) -> numpy.array:
    """
    Given a vector, typically representing the direction something
    is moving, calculate a unit vector representing an angle based
    off the aspect of the origin vector.

    This is done by reversing the direction vector, to establish the
    reference point, then finding its angle in global space.

    That angle then has the offset applied, and returned.

    """

    opposite = -1 * direction

    angle_rad = numpy.arctan2(opposite[1], opposite[0])
    offset_rad = numpy.deg2rad(offset_deg)

    new_angle = angle_rad + offset_rad

    if new_angle > 2 * numpy.pi:
        new_angle -= 2 * numpy.pi
    elif new_angle < 0:
        new_angle += 2 * numpy.pi

    offset_unit_vec = numpy.array(
        [numpy.cos(new_angle), numpy.sin(new_angle), 0]
    )

    return offset_unit_vec


def calculate_antenna_train_angle(
    heading_unit_vec: numpy.array, relative_pos_unit_vec: numpy.array
) -> float:
    """
    Given the direction of an object, and a unit vector representing
    the direction towards an object, find the angle between those objects
    relative to the first one's direction.

    This is done by subtracting out the heading from the relative position,
    to keep things in reference to originator. Divide by the magnitude
    to keep as a unit vector, then find angle between.

    """

    diff = relative_pos_unit_vec - heading_unit_vec

    diff /= magnitude(diff)

    return numpy.arctan2(diff[1], diff[0])


def calculate_unit_vec_between(vec: numpy.array, vec_2: numpy.array):
    """Return a unit vector between two points"""
    return (vec_2 - vec) / magnitude(vec_2 - vec)


def create_intercept_location(
    interceptor: Particle,
    target: Particle,
    interceptor_speed: float,
    target_speed: float,
) -> numpy.array:

    location, _ = calculate_2d_intercept(
        interceptor.position,
        target.position,
        target.behavior.end,
        interceptor_speed,
        target_speed,
    )

    return location
