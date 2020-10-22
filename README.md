# kthackaton

A simple recommandation system for movies, developed for the KTH Hackaton
organized by [KTHAIS](https://kthais.com/).

It will generate a tree of suggestions starting from the movie provided as
input, which you can then visualize by opening `index.html` in a browser. 

Highly inspired by [this notebook on kaggle](https://www.kaggle.com/fabiendaniel/film-recommendation-engine).

## How to 

You need first of all to download the movie database, used by the system, which can be
found [here](https://www.kaggle.com/rounakbanik/the-movies-dataset). Extract it
into the `input` folder. Your directories should look like this


```
input/
    tmdb-movie-metadata/
        ...
    keywords.csv
    credits.cvs
    ...
index.html
index.js
...
```

Then install the needed dependencies listed in the requirements file 

```shell
$ pip install -r requirements.txt
```

After that just run `python main.py`, it will start loading the needed files
and initializing the system. After that it will prompt you for entering the
film you want to be recommended on. This will generate a `tree.js` file which
will contain the informations about the recommendations, as well as downloading
the image associated to the suggested films.

Once the process ends just open `index.html` in your favourite browser and explore the graph (sorry for the
bad-looking page, if you want to spend some time tweaking the `css` I would be
very grateful, just open a PR)!
