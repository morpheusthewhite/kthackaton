from recommandation.recommend import recommend_by_title
from recommandation.poster import fetch_images
from recommandation.moviedb import info_movie


def build_recommandation_tree(movie_title, depth, movies):

    recommandations = recommend_by_title(movie_title)
    recommandations_unique = list(filter(lambda elem: elem not in movies, recommandations))

    if depth == 0:
        for recommandation in recommandations_unique:
            # print(info_movie(recommandation))
            build_recommandation_tree(recommandations, depth-1, movies + recommandations_unique)


def build_tree(movie_title, depth):
    movies = [movie_title]

    root_info = info_movie(movie_title)
    filename = fetch_images(movie_title)
    tree = root_info
    tree["filename"] = filename
    parents = [root_info]

    for i in range(0, depth):

        next_parents = []
        for parent in parents:
            current_movie = parent['name']
            recommandations = recommend_by_title(current_movie)
            for recommandation in recommandations:

                # the recommandation is not a duplicate
                if recommandation not in movies:
                    # append it to the list of met movies
                    movies.append(recommandation)

                    filename = fetch_images(recommandation)

                    # retrieve the info related to the current recommendation
                    recommandation_complete = info_movie(recommandation)
                    recommandation_complete["filename"] = filename

                    # append it to the list of recommendations of the parent
                    parent['children'] = parent.get('children', []) + [recommandation_complete]

                    next_parents.append(recommandation_complete)

        parents = next_parents

    return tree
