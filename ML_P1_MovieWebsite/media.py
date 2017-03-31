import webbrowser

# defind class Movie
class Movie():
    # movie parameters
    def __init__(self,movie_title,poster_image,tralier_youtube):
        self.title = movie_title
        self.poster_image_url = poster_image
        self.trailer_youtube_url = tralier_youtube

    def show_trailer(self):
        webbrowser.open(self.trailer_youtube_url)
