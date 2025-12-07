#!/usr/bin/env python3
"""safsfafsa"""


def insert_school(mongo_collection, **kwargs):
    """
    asfasfasfas
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
