from recommandation.moviedb import info_movie

def filter_movie_list( list_movie, dict_filter):
    res=[]
    for i in range (0,len(list_movie)):
        movie = list_movie[i]
        dict_movie = info_movie(movie)
        if (dict_movie["director"] not in dict_filter["director"]
            or dict_movie["runtime"] > dict_filter["runtime"]
            or dict_filter["genre"] not in dict_movie["genre"]
            or dict_filter["actor"] not in dict_movie["actor"]
            or date_to_int(dict_movie["release_date"])<date_to_int(dict_filter["release_min"])
            or date_to_int(dict_movie["release_date"])>date_to_int(dict_filter["release_max"])
            ):
            print("refuse")
        else:
            print("admit")
            res.append(movie)
    return(res)


def date_to_int(date):
    result =0
    for i in range(0,len(date)):
        if date[i]!='-':
            result=result*10+int(date[i])
    return (result)

