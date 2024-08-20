"""import subprocess

p = subprocess.Popen('ls -l', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in p.stdout.readlines():
    print(line)
retval = p.wait()"""

import requests

weight = 4.0

entity_a = {
    "_id": 20,
    "profile_id": "Match1436",
    "starring": "Kem Sereyvuth",
    "title": "City of Ghosts",
    "writer": "Mike Jones (screenwriter)"
}

entity_b = {
    "_id": 20,
    "profile_id": "Match1436",
    "year": "2002",
    "director_name": "Dillon, Matt (I)",
    "genre": "Crime",
    "imdb_ksearch_id": "4684",
    "title": "City of (2002)"
}

entity_pair = {'entity_a': entity_a,
               'entity_b': entity_b,
               'weight': weight
}

response = requests.post("http://127.0.0.1:5000/predict", json=entity_pair)


print(response.text)