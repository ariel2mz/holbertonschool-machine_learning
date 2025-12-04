#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import requests


def availableShips(passengerCount):
    """
    kaklsksaklsl aklslka
    """
    url = "https://swapi.dev/api/starships/"
    ships_list = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            pas = ship.get('passengers', '0')

            if pas == 'unknown' or pas == 'n/a' or not pas:
                continue

            pas = pas.replace(',', '')

            try:
                if int(pas) >= passengerCount:
                    ships_list.append(ship['name'])
            except ValueError:
                continue

        url = data.get('next')

    return ships_list
