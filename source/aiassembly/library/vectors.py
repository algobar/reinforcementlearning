import math
import numpy
from typing import Tuple

ABSOLUTE_NORTH: numpy.ndarray = numpy.ndarray([0, 1, 0], dtype=numpy.float32)
Degrees = float


def copy_array(arr: numpy.ndarray) -> numpy.ndarray:
    """Return a copy of the array"""
    return numpy.copy(arr)


def magnitude(vec: numpy.ndarray) -> float:
    """Find the magnitude of the vector

    Args:
        vec (numpy.ndarray): [description]

    Returns:
        float: scalar representing magnitude
    """

    return numpy.linalg.norm(vec, axis=-1)


def get_unit_vector(array: numpy.ndarray) -> numpy.ndarray:
    """Return the unit vector of the array"""
    return array / magnitude(array)


def calculate_absolute_bearing(
    origin: numpy.ndarray, destination: numpy.ndarray
) -> float:
    """Calculate the bearing from the origin to the destination. Note that
    with the particle designation, this does not include any heading specific
    information, assuming that the bearing angle being calculated from the
    y axis.

    Args:
        origin (numpy.ndarray): [description]
        destination (numpy.ndarray): [description]

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
    interceptor_pos: numpy.ndarray,
    target_pos: numpy.ndarray,
    target_dest: numpy.ndarray,
    interceptor_speed: float,
    target_speed: float,
) -> Tuple[numpy.ndarray, float]:
    """

    Args:
        interceptor_pos (numpy.ndarray): [description]
        target_pos (numpy.ndarray): [description]
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

    AC: numpy.ndarray = interceptor_pos - target_pos
    AT: numpy.ndarray = target_dest - target_pos

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
    current: numpy.ndarray, end: numpy.ndarray, speed: float, timestep: float
) -> numpy.ndarray:
    """Calculates the updated position based on the desired end position"""
    delta_pos: numpy.ndarray = end - current

    hypotenuse: float = magnitude(delta_pos)

    angle: float = math.atan2(delta_pos[1], delta_pos[0])

    step_size: float = speed * timestep

    if step_size > hypotenuse:
        step_size = hypotenuse

    delta_x: float = step_size * math.cos(angle)
    delta_y: float = step_size * math.sin(angle)

    delta_arr = numpy.ndarray([delta_x, delta_y, 0], dtype=current.dtype)
    current += delta_arr

    return current


def calculate_aspect(
    originator_heading: numpy.ndarray,
    originator_position: numpy.ndarray,
    target_position: numpy.ndarray,
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
    direction: numpy.ndarray, offset_deg: float
) -> numpy.ndarray:
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

    offset_unit_vec = numpy.ndarray(
        [numpy.cos(new_angle), numpy.sin(new_angle), 0]
    )

    return offset_unit_vec


def calculate_antenna_train_angle(
    heading_unit_vec: numpy.ndarray, relative_pos_unit_vec: numpy.ndarray
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


def calculate_unit_vec_between(vec: numpy.ndarray, vec_2: numpy.ndarray):
    """Return a unit vector between two points"""
    return (vec_2 - vec) / magnitude(vec_2 - vec)


def create_intercept_location(
    interceptor_position: numpy.ndarray,
    target_position: numpy.ndarray,
    interceptor_speed: numpy.ndarray,
    target_speed: numpy.ndarray,
) -> numpy.ndarray:

    # TODO create a point extrapolated
    # target current 
    target_dest = None

    location, _ = calculate_2d_intercept(
        interceptor_position,
        target_position,
        target_dest,
        interceptor_speed,
        target_speed,
    )

    return location
