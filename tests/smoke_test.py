"""Smoke-test a running API using only the Python standard library."""
from http.client import HTTPException
import json
import math
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
EXAMPLE = {
    "trans_date_trans_time": "2019-01-01T00:00:18",
    "amt": 4.97, "category": "misc_net", "gender": "F",
    "dob": "1988-03-09", "lat": 36.0788, "long": -81.1781,
    "merch_lat": 36.011293, "merch_long": -82.048315, "city_pop": 3495,
}

def request(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    for attempt in range(60):
        try:
            status, health = request("/health")
            if status == 200 and health == {"status": "ok"}:
                break
        except (URLError, TimeoutError, ConnectionError, HTTPException):
            pass
        time.sleep(2)
    else:
        raise AssertionError("API did not become healthy within 120 seconds")

    status, result = request("/score", EXAMPLE)
    assert status == 200, (status, result)
    assert math.isclose(result["fraud_probability"], 0.000022, abs_tol=0.000002), result
    assert result["is_fraud"] is False, result
    assert math.isclose(result["threshold"], 0.3482, abs_tol=0.0001), result

    changed = dict(EXAMPLE, amt=950, category="shopping_net", trans_date_trans_time="2019-01-01T23:15:00")
    status, higher = request("/score", changed)
    assert status == 200 and higher["fraud_probability"] > result["fraud_probability"], higher

    for patch in ({"amt": -1}, {"category": "unknown"}, {"lat": 91}, {"gender": "X"},
                  {"trans_date_trans_time": "2019-01-01T00:00:18Z"}):
        status, error = request("/score", dict(EXAMPLE, **patch))
        assert status == 422, (patch, status, error)
    status, error = request("/score", {})
    assert status == 422, (status, error)
    print("PASS: health, baseline score, increased-risk example, and six invalid inputs")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
