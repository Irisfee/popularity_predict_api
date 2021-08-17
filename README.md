# Popularity prediction model API
This model is for predicting the popularity of Mandopop/C-pop song. You can simply upload a song (a clip of 30s that can highlight the song works best), and the model will tell you the estimated popularity of it.

The popularity metric is defined in [Spotify](https://developer.spotify.com/documentation/web-api/reference/#objects-index). The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.
Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past.

# Model description
## Structure
The model converts the song into mel-spectrogram and auto-tags the song with a CNN designed for making accurate frame-level predictions of tags in music clips[2]. Then both the mel-spectrogram and auto-tag will pass through another CNN designed for song prediction [1].
The output of the model will be a simple popularity score.

## Training data
We scaped the preview version of all songs belong to the mandopop genre on Spotify and train the model with these songs.

# Usage
To use this API, 
1. clone the repo and download the python module in the requirement.txt. 
2. run app.py
3. type http://127.0.0.1:5000/ and upload the mp3 file in the webpage.
4. wait about 20s and the predicted score will be on the screen!
We will wrap these steps up with a docker container soon!

# Reference
[1] Yu, L. C., Yang, Y. H., Hung, Y. N., & Chen, Y. A. (2017). Hit song prediction for pop music by siamese cnn with ranking loss. arXiv preprint arXiv:1710.10814. [repo](https://github.com/OckhamsRazor/HSP_CNN)

[2] Liu, J. Y., & Yang, Y. H. (2016, October). Event localization in music auto-tagging. In Proceedings of the 24th ACM international conference on Multimedia (pp. 1048-1057).
