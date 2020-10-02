import tmdbsimple as tmdb

tmdb.API_KEY = 'eb6b2d276dcdd4817ef758cdeb679b1c'

search = tmdb.Search()
response = search.movie(query='Blade Runner')
for s in search.results:
    print(s['title'])
    print(s)

print("  ")
print(search)
