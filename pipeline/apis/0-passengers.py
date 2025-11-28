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
            passengers = ship.get('passengers', '0')
            
            if passengers == 'unknown' or passengers == 'n/a' or not passengers:
                continue
            
            passengers = passengers.replace(',', '')
            
            try:
                if int(passengers) >= passengerCount:
                    ships_list.append(ship['name'])
            except ValueError:
                continue
        
        url = data.get('next')
    
    return ships_list
