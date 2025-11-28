#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import requests


def sentientPlanets():
    """
    asfsafsa
    """
    surl = "https://swapi.dev/api/species/"
    pset = set()

    while surl:
        response = requests.get(surl)
        data = response.json()

        for species in data['results']:
            clas = species.get('classification', '').lower()
            desi = species.get('designation', '').lower()

            if 'sentient' in clas or 'sentient' in desi:
                homeworld = species.get('homeworld')
                if homeworld:
                    pres = requests.get(homeworld)
                    pdata = pres.json()
                    pset.add(pdata['name'])

        surl = data.get('next')

    return list(pset)
