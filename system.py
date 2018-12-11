"""OCR Assignment
Written by: Catalin Mares
version: v1.0
"""
from scipy import ndimage
import numpy as np
import utils.utils as utils
import scipy.linalg
from sklearn.metrics import mean_squared_error
from collections import Counter


def reduce_dimensions(feature_vectors_full, model):
    """Reducing to 10 dimensions using PCA

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    computed_v = model['computed_v']
    pca_data = np.dot(
        (feature_vectors_full - np.mean(feature_vectors_full)), computed_v)
    return pca_data


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.
    The eigenvalues are computed here on the training data to be used
    for the PCA dimension reduction. The same V value is stored in the
    model so that it is reused again on the training data and not being
    recomputed again.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)
    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    covx = np.cov(fvectors_train_full, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1))
    computed_v = np.fliplr(v)
    model_data['computed_v'] = computed_v.tolist()
    # reading the words from the dictionary
    model_data['dictionary_words'] = [word for line in open(
        "wordsEn.txt", 'r') for word in line.split()]
    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix. Also
    as the noise on the pages is salt and pepper noise, a median
    filter is also applied to the test data to reduce some of the
    noise. An attempt of noise level detection has been attempted
    but not successful. This was going to be done in order to
    tune the KNN nearest neighbour according to the noise, so that
    the more noise on the page the bigger the KNN value.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    # For every row in images_test,reduce the noise
    reduced_noise = list(map(noise_reduction, images_test))

    # Tried working out the noise
    # count=0
    # for i in range (len(images_test)):
    #     for x in range (len(images_test[i])):
    #         if(images_test[i][x].shape != (0,) or reduced_noise[i][x].shape != (0,)):
    #             count=count+mean_squared_error(reduced_noise[i][x], images_test[i][x])
    # print(count)

    fvectors_test = images_to_feature_vectors(reduced_noise, bbox_size)

    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def noise_reduction(page_image):
    """Applying a median filter to each page. This is because the type
    of noise on the pages is salt and pepper noise which is best removed
    using a median filter.
    """
    return ndimage.median_filter(page_image, 3)


def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    for x in range(len(labels)):
        if ((x != 0) and (len(labels) >= (x+2)) and (labels[x] == 'l') and
                ((labels[x+1] == 'r') and (labels[x+2] == 'l'))):
            labels[x] = "'"

    # Sorting out the apostrophe for 're words. Only if the word is not
    # 'already'. Apostrophe is misclassified with l sometimes
    for x in range(len(labels)):
        if ((x != 0) and (len(labels) >= (x+2)) and (labels[x-1] != 'a') and
                (labels[x] == 'l') and ((labels[x+1] == 'r') and
                                        (labels[x+2] == 'e'))):
            labels[x] = "'"
    # nlt will be changed to n't in case don't is found
    for x in range(len(labels)):
        if ((x != 0) and (len(labels) >= (x+2)) and (labels[x] == 'l') and
                ((labels[x+1] == 't') and (labels[x-1] == 'n'))):
            labels[x] = "'"
    # Ilm is changed to I'm so that it corrects some errors
    for x in range(len(labels)):
        if ((x != 0) and (len(labels) >= (x+2)) and (labels[x] == 'l') and
                ((labels[x-1] == 'I') and (labels[x+1] == 'm'))):
            labels[x] = "'"

    # reading the words from the dictionary
    # dictionary_words = model['dictionary_words']

    # Finding out the indices where the spacing is between words
    # word_ends = []
    # for x in range (len(bboxes)-1):
    #     if (((bboxes[x+1][0] - bboxes[x][2])>6)):
    #         word_ends.append(x+1)

    '''Making every word that is found on the page 
    using the indices where the spaces have been found 
    to join together the correct characters that make up the words
    '''
    # words = []
    # for x in range(len(word_ends)):
    #     if (x==0):
    #         words.append(''.join(labels[x:(word_ends[x])]))
    #     else:
    #         words.append(''.join(labels[((word_ends[x-1])):(word_ends[x])]))

    '''Finding out the indexes of the words that do not require to be modified
    as they exist in the dictionary
    '''
    # index_of_words = []
    # for y in range(len(words)):
    #     for x in range(len(dictionary_words)):
    #         if ((words[y] == dictionary_words[x])):
    #             index_of_words.append(y)

    '''Here I am looking in the dictionary only for the words that have not been
    matched in the dictionary in the code above. The word is corrected only if a
    word is found in the dictionary that has a difference of 1 between the original
    word and the word found in the dictionary.
    '''
    # for y in range(len(words)):
    #     if(y not in index_of_words):
    #         for x in range(len(dictionary_words)):
    #             if((len(words[y])>3) and (words[y] != dictionary_words[x]) and
    #                 ((len(words[y])) == (len(dictionary_words[x]))) and
    #                     ((compute_diff(words[y],dictionary_words[x]))==1)):
    #                         words[y]=dictionary_words[x]

    # shape does not match, I have lost some labels in the process of correcting errors
    # modified_labels = np.asarray(list("".join(words)))

    return labels


def compute_diff(s1, s2):
    """Function to compute the difference between two strings

    parameters:

    s1 - the first string to be compared
    s2 - the second string to be compared
    """

    return sum(1 for a, b in zip(s1, s2) if a != b)


def classify_page(page, model):
    """KNN classifier for the first 9 nearest neighbours

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    x = np.dot(page, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(page * page, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)

    return k_nearest_neighbour(12, dist, labels_train, nearest, page)


def k_nearest_neighbour(k, dist, labels_train, nearest, page):
    """This function returns the most occurring K nearest neighbours.

    parameters:

    k - int, number of K nearest neighbours to be computed
    dist - a matrix with the cosine distances between the labels
    labels_train = all the training labels
    nearest = a matrix containing the indices of the closest neighbours
    page - matrix, each row is a feature vector to be classified
    """
    nearest_neighbours = np.zeros((k, len(page)))
    labels = np.empty((k, len(page)), type('r'))

    for i in range(k):
        nearest_neighbours[i] = nearest
        labels[i] = labels_train[nearest]
        for x in range(len(page)):
            dist[x][nearest[x]] = 0
        nearest = np.argmax(dist, axis=1)

    for i in range(labels.shape[1]):
        count = Counter(labels[:, i])
        labels[0][i] = count.most_common(1)[0][0]

    return labels[0]
