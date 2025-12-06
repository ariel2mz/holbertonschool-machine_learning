#!/usr/bin/env python3
"""safasfasfsafs"""


def update_topics(mongo_collection, name, topics):
    """
    safasfsafsa
    fsafasfasfs
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
