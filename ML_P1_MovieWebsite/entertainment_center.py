import media
import fresh_tomatoes

iron_man = media.Movie("Iron Man",
                       "Tony Stack become the Iron Man",
                       "https://img.unlonecdn.ru/2016/06/27/poster/fd41bd695a884395145347399e62d4d7-iron-man-1467058232.jpg",
                       "https://www.youtube.com/watch?v=8hYlB38asDY")

iron_man2 = media.Movie("Iron Man 2",
                       "Iron Man fight with War Machine",
                       "https://s-media-cache-ak0.pinimg.com/originals/f1/6d/c6/f16dc6f627ae65879a2720282d024965.jpg",
                       "https://www.youtube.com/watch?v=BoohRoVA9WQ")

iron_man3 = media.Movie("Iron Man 3",
                       "Iron Man's final fight",
                       "https://images-na.ssl-images-amazon.com/images/M/MV5BMTkzMjEzMjY1M15BMl5BanBnXkFtZTcwNTMxOTYyOQ@@._V1_UY1200_CR105,0,630,1200_AL_.jpg",
                       "https://www.youtube.com/watch?v=Ke1Y3P9D0Bc")

captain_america = media.Movie("Captain America : The First Avenger",
                       "Steven Rogers become the Captain America",
                       "https://static5.comicvine.com/uploads/original/0/40/1885267-tumblr_lnm9k6scjt1qzniqdo1_1280.jpeg",
                       "https://www.youtube.com/watch?v=JerVrbLldXw")

captain_america2 = media.Movie("Captain America : Winter Soldier",
                       "Fallen down of SHIELD",
                       "http://vignette2.wikia.nocookie.net/marvelcinematicuniverse/images/2/26/Cap_2_poster.jpg/revision/latest?cb=20140131142227",
                       "https://www.youtube.com/watch?v=7SlILk2WMTI")

captain_america3 = media.Movie("Captain America : Civil War",
                       "Captain America vs Iron Man",
                       "http://cdn3-www.comingsoon.net/assets/uploads/gallery/captain-america-3-1413251820/cpkf2a1.jpg",
                       "https://www.youtube.com/watch?v=dKrVegVI0Us")

#print iron_man3.stroyline
#captain_america3.show_trailer()
my_movies = [iron_man,iron_man2,iron_man3,captain_america,captain_america2,captain_america3]
fresh_tomatoes.open_movies_page(my_movies)