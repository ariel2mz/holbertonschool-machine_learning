#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)

    url = sys.argv[1]

    try:
        response = requests.get(url)

        if response.status_code == 403:
            reset_timestamp = int(response.headers.get('X-Ratelimit-Reset', 0))
            current_timestamp = int(datetime.now().timestamp())
            minutes_remaining = (reset_timestamp - current_timestamp) // 60
            print(f"Reset in {minutes_remaining} min")
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            print(location if location else "")
        else:
            print("Not found")

    except Exception:
        print("Not found")
