import webbrowser

# Defind class Movie
class Movie():
    """ Constructor for init movie parameters
        Paras : pointer,movie attributes"""
    def __init__(self,movie_title,poster_image,tralier_youtube):
        self.title = movie_title
        self.poster_image_url = poster_image
        self.trailer_youtube_url = tralier_youtube

    """ Show current movie instance's trailer in browser
        Paras : pointer"""
    def show_trailer(self):
        webbrowser.open(self.trailer_youtube_url)
