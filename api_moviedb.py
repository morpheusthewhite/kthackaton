import tmdbsimple as tmdb

name = 'Kick-Ass 2'

def info_movie(name):

    tmdb.API_KEY = 'eb6b2d276dcdd4817ef758cdeb679b1c'
    search = tmdb.Search()

    response = search.movie(query=name)

    for elt in response["results"]:
        print(elt["title"])
        if elt["title"]== name:
            s = elt

    movie = tmdb.Movies(s["id"])
    print("\n\n")

    actor=[]
    director=[]
    runtime=0
    release_date=0
    reponse = movie.info()
    runtime = movie.runtime
    genres=[]
    release_date = reponse["release_date"]
    for elt in reponse["genres"]:
        genres.append(elt["name"])

    reponse = movie.credits()

    for truc in reponse["cast"]:
        actor.append(truc["name"])
    for truc in reponse["crew"]:
        if truc["job"]=="Director":
            director = truc["name"]

    dictionnaire={}
    dictionnaire["director"]=director
    dictionnaire["runtime"]=runtime
    dictionnaire["release_date"]=release_date
    dictionnaire["actor"]=actor
    dictionnaire["genres"]=genres

    return(dictionnaire)











