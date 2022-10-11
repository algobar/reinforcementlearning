"""
Module to define features and the composition
of features into branches and collections

End goal: 

1. State can be represented in a type
enforced manner. We can track the features
in a tree represented manner if there 
exists a many to one feature set.

2. A sim agnostic structure provides means
of requesting features, and having the simulation
provide the data as needed

State can be squashed into a flat array
where it can easily be passed to different
functions to perform matrix based operations
where applicable. 

Focus on pulling only raw data...
Remove calculations that can be done at 
the extraction level when it comes
to relationship specific data
or data over time (unless specified)


if we look at a tree structure,
we can organize things into (3) categories
- feature collection - represents one or more feature branch
- feature branch - represents one or more features
- features - float/int representing data

The important part is that we understand that 
we don't leave dangling features outside of a branch.


To define this groups, the gym environment
needs to list out the properties to measure.

{
    "structure":"feature_collection"
    "selection": "group",
    "selection_config": {
        "side": "agent"
    }
    "data":
    [
        {
            "structure": "feature branch"
            "features":
            [
                {
                    "name": "position_x"
                },
                {
                    "name": "position_y"
                },
                {
                    "name": "position_z
                }

            ]
        },
        {
            "structure": "feature_collection",
            "selection": "sensed"
            "selection_config":{
                "side": "enemy
            },
                "data":
            [
                {
                    "structure": "feature branch"
                    "features":
                    [
                        {
                            "name": "position_x"
                        },
                        {
                            "name": "position_y"
                        },
                        {
                            "name": "position_z
                        }

                    ]
                }
            ]
    ]
}

Usage: 
1. This config is created by the user
for data to 'request' from the environment.
2. Simulation takes this config and ensures
all features can be met
3. At state collection step, simulation
will extract data based on the requested
structure shown here

The root level corresponds to an entity
type. This type definition is loose at 
the moment. But can separate data based
on different object types: (i.e car vs
airplane).

Based on the grouping above, for each
type where there exists a repeated 
relationship, would drive multiple
resulting feature branches



"""
from aiassembly.environments.types import (
    Feature,
    FeatureBranch,
    FeatureCollection,
    FeatureBranchRepr,
)


def initialize_feature_branch() -> FeatureBranch:
    """Creates a new feature branch"""
    return []


def initialize_feature_collection() -> FeatureCollection:
    """Creates a new feature collection"""
    return []


def add_to_feature_branch(
    feature: Feature, feature_branch: FeatureBranch
) -> FeatureBranch:
    """Adds a new feature to the given branch"""
    feature_branch.append(feature)


def add_to_feature_collection(
    feature_branch: FeatureBranch, feature_collection: FeatureCollection
) -> FeatureCollection:
    """Adds feature branch to collection"""
    feature_collection.append(feature_branch)
