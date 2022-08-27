"""
Module to define features and the composition
of features into branches and collections

"""
from aiassembly.environments.types import Feature, FeatureBranch, FeatureCollection


class PositionFeatures:

    position_x = "position_x"
    position_y = "position_y"
    position_z = "position_z"


class VelocityFeatures:
    speed = "speed"


def initialize_feature_branch() -> FeatureBranch:

    return []


def initialize_feature_collection() -> FeatureCollection:

    return []


def add_to_feature_branch(
    feature: Feature, feature_branch: FeatureBranch
) -> FeatureBranch:

    feature_branch.append(feature)


def add_to_feature_collection(
    feature_branch: FeatureBranch, feature_collection: FeatureCollection
) -> FeatureCollection:

    feature_collection.append(feature_branch)
