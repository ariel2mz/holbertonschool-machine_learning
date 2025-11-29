#!/usr/bin/env python3
"""
SpaceX launch frequency by rocket
"""
import requests
import sys


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"

    try:
        response = requests.get(url)
        launches = response.json()

        rocket_counts = {}

        for launch in launches:
            rocket_id = launch['rocket']
            rocket_url = (
                f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
            rocket_response = requests.get(rocket_url)
            rocket_data = rocket_response.json()
            rocket_name = rocket_data['name']

            current_count = rocket_counts.get(rocket_name, 0)
            rocket_counts[rocket_name] = current_count + 1

        sorted_rockets = sorted(
            rocket_counts.items(), key=lambda x: (-x[1], x[0]))

        for rocket_name, count in sorted_rockets:
            print(f"{rocket_name}: {count}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
