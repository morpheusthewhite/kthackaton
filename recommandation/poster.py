# importing the module
import imdb
import requests
from os.path import join


CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
KEY = 'eb6b2d276dcdd4817ef758cdeb679b1c'

url = CONFIG_PATTERN.format(key=KEY)
r = requests.get(url)
config = r.json()

# creating instance of IMDb
ia = imdb.IMDb()


# name
def fetch_images(name):
    # searching the name
    search = ia.search_movie(name)
    id = "tt" + search[0].movieID
    # print(id)
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
    if "status_message" in api_response.keys():
        if api_response["status_message"] == "The resource you requested could not be found.":
            # print("no image :(")
            return ""

    posters = api_response['posters']
    poster_urls = []
    for poster in posters:
        rel_path = poster['file_path']
        url = "{0}{1}{2}".format(base_url, max_size, rel_path)
        poster_urls.append(url)

    r = requests.get(poster_urls[0])
    filetype = r.headers['content-type'].split('/')[-1]
    filename = name + ".{0}".format(filetype)
    with open(join("img", filename), 'wb') as w:
        w.write(r.content)

    return filename