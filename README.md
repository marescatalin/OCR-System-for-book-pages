# OCR-System-for-book-pages

## Feature Extraction
I have chosen to reduce the dimentions of my data using Principal
Component Analysis (PCA). I have done this because I believe that
the letters have quite a bit of variation across classes. This 
dimention reduction was done by firstly I have obtained the
Eigenvalues and Eigenvectors from the covariance matrix and have
sorted these to extract the 10 eigenvectors that correspond to the
10 largest eigenvalues. I have used these to construct the projection
matrix v and then transformed the original dataset using v to obtain
a 10-dimentional feauture space.
## Classifier 
For the classifier I have firstly used a basic K nearest neighbour
classifier in order to classify the labels according to the first
nearest neighbour using cosine distance, picking out the largest
cosine distance as this is a measure of similarity. I have also 
experimented using euclidian distance, a measure of disimilarity 
but with worse performance. I have also implemented a kNN nearest
neighbour classifier. This was implemented successfully with performance
increasing significantly for the pages containing more noise.I believe
kNN was a good choice because it is not model based and has a low bias.
kNN value of 12 seemed to work the best.
## Error Correction
First of all I have noticed that quite a lot of apostrophies have been
missclassified for the letter "l". Therefore I have researched online
the most common scenarios where this occurs. I have attempted to correct
a few errors such as "nlt" being missclassified for "n't" and a few more 
similar cases. This gave a slight score improvement. After that I have 
tried using a dictionary in order to correct some of the words collected 
from the page. I have used the bounding boxes coordinates to work out the
horizontal distance between the bounding boxes and find where the spaces 
between the words were by comparing each horizontal distance to a certain 
threshold value. I have then extracted the words and compared them with words
from a dictionary. If the word was not found in the dictionary, an attempt at
correction would be made by taking the closest match (character difference of 1)
and (same length of characters). Unfortunately a few labels have been lost
throughout this process therefore I was not able to send the new labels through
to the classifier as the shapes did not quite match exactly. The code has been 
left in the function but it has been commented out. 
## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
Page 1: score = 95.6% correct
Page 2: score = 96.1% correct
Page 3: score = 89.2% correct
Page 4: score = 76.4% correct
Page 5: score = 63.5% correct
Page 6: score = 50.3% correct
## Other information 
The noise found on the pages was salt and pepper noise. This is best filtered using a median
filter.This has been implemented giving significant performance improvements.
Also in order to tune my kNN classifier I have tried to work out the level of noise
on each page to choose a higher kNN value the higher the noise found on the page but
this was not 100% as I could not find a correct comparison between the matrix before and
after the noise reduction was done. The code has been left in the file but commented out.

