#!/usr/bin/env python3
"""
SpaceX upcoming launch information
"""
import requests
import sys


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"

    try:
        response = requests.get(url)
        launches = response.json()

        sorted_launches = sorted(launches, key=lambda x: x['date_unix'])

        for launch in sorted_launches:
            if launch['upcoming']:
                launch_name = launch['name']
                date_local = launch['date_local']

                rid = launch['rocket']
                ru = f"https://api.spacexdata.com/v4/rockets/{rid}"
                rocket_response = requests.get(ru)
                rocket_data = rocket_response.json()
                rocket_name = rocket_data['name']

                lid = launch['launchpad']
                lu = f"https://api.spacexdata.com/v4/launchpads/{lid}"
                launchpad_response = requests.get(lu)
                launchpad_data = launchpad_response.json()
                launchpad_name = launchpad_data['name']
                launchpad_locality = launchpad_data['locality']

                output = (f"{launch_name} ({date_local}) {rocket_name} - "
                          f"{launchpad_name} ({launchpad_locality})")
                print(output)
                break

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
