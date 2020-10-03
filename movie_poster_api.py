# importing the module
import imdb
import requests
CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
KEY = 'eb6b2d276dcdd4817ef758cdeb679b1c'

url = CONFIG_PATTERN.format(key=KEY)
r = requests.get(url)
config = r.json()

# creating instance of IMDb
ia = imdb.IMDb()

# name
name = "bladerunner"

# searching the name
search = ia.search_movie(name)
print(search)
id = "tt" + search[0].movieID
print(id)
base_url = config['images']['base_url']
sizes = config['images']['poster_sizes']
"""
    'sizes' should be sorted in ascending order, so
        max_size = sizes[-1]
    should get the largest size as well.        
"""
def size_str_to_int(x):
    return float("inf") if x == 'original' else int(x[1:])
max_size = max(sizes, key=size_str_to_int)

IMG_PATTERN = 'http://api.themoviedb.org/3/movie/{imdbid}/images?api_key={key}'
r = requests.get(IMG_PATTERN.format(key=KEY,imdbid=id))
api_response = r.json()
print(api_response)
if "status_message" in api_response.keys():
    if api_response["status_message"] == "The resource you requested could not be found.":
        print("no image :(")
        exit(0)
#rel_path =api_response['backdrops'][0]['file_path'][1:]
#print(api_response['backdrops'][0]['file_path'][1:])
#url = base_url + max_size + rel_path
#print(url)
#if api_response["status_message"].exists():
#    if api_response["status_message"] == "The resource you requested could not be found.":
#        print("no image :(")
#        exit(0)

posters = api_response['posters']
poster_urls = []
for poster in posters:
    rel_path = poster['file_path']
    url = "{0}{1}{2}".format(base_url, max_size, rel_path)
    poster_urls.append(url)
print(poster_urls)
#for nr, url in enumerate(poster_urls):
#    r = requests.get(url)
#    filetype = r.headers['content-type'].split('/')[-1]
#    filename = 'poster_{0}.{1}'.format(nr+1,filetype)
#    with open(filename,'wb') as w:
#        w.write(r.content)

poster_urls[0]
r = requests.get(poster_urls[0])
filetype = r.headers['content-type'].split('/')[-1]
filename = name+ ".{0}".format(filetype)
with open(filename, 'wb') as w:
    w.write(r.content)