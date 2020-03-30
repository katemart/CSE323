'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.
    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.
    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # TODO: Start sampling (12 points)
    # ediff1d gives difference between consecutive elements of the array
    # we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    # basically when words like mm or ii have no path / little path, use centroid
    if (distance[-1] == 0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        # get the proportion of line segments
        distance = distance / distance[-1]
        # scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        # generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        # use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()
    return sample_points_X, sample_points_Y


# function to calculate distance between two points
def distance(x1, y1, x2, y2):
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5


# function to normalize points
def normalize(sample_points_X, sample_points_Y, L=50):
    # get s value for every point
    width = max(sample_points_X) - min(sample_points_X)
    height = max(sample_points_Y) - min(sample_points_Y)
    if width == 0 and height == 0:
        s = 0
    else:
        s = L / max(width, height)
    # scale and translate the points
    scaled_points = list(s * elem for elem in sample_points_X), list(s * elem for elem in sample_points_Y)
    x_centroid, y_centroid = np.mean(scaled_points[0]), np.mean(scaled_points[1])
    px, py = np.reshape((0 - x_centroid), (-1, 1)), np.reshape((0 - y_centroid), (-1, 1))
    norm_points = px + py + scaled_points
    return norm_points


# Pre-sample every template (and normalize temp sample points at the same time)
template_sample_points_X, template_sample_points_Y = [], []
norm_temp_sample_points_X, norm_temp_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    # cache normalized template points (to be used later)
    norm_x, norm_y = normalize(X, Y)
    norm_temp_sample_points_X.append(norm_x)
    norm_temp_sample_points_Y.append(norm_y)
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


valid_temp_sample_point_index = []
def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.
    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    global valid_temp_sample_point_index
    valid_temp_sample_point_index = []
    # TODO: Set your own pruning threshold
    threshold = 15
    # TODO: Do pruning (12 points)
    # get distance between each gesture and template point
    for i in range(10000):
        start_distance = distance(gesture_points_X[0], gesture_points_Y[0], norm_temp_sample_points_X[i][0], norm_temp_sample_points_Y[i][0])
        end_distance = distance(gesture_points_X[-1], gesture_points_Y[-1], norm_temp_sample_points_X[i][-1], norm_temp_sample_points_Y[i][-1])
        if start_distance <= threshold and end_distance <= threshold:
            valid_words.append(words[i])
            valid_temp_sample_point_index.append(i)
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    # normalized points are passed in as indices (for efficiency purposes)
    L = 50  # defined as default value in normalize function to improve performance (so not needed here)
    # TODO: Calculate shape scores (12 points)
    # get distance between each gesture and template point
    global valid_temp_sample_point_index
    for i in range(len(valid_template_sample_points_X)):
        distances = 0
        index = valid_temp_sample_point_index[i]
        valid_template_sample_points_X[i], valid_template_sample_points_Y[i] = norm_temp_sample_points_X[index], norm_temp_sample_points_Y[index]
        # number of sample points is 100
        for j in range(100):
            dist = distance(valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j],
                            gesture_sample_points_X[j], gesture_sample_points_Y[j])
            distances += dist
        # set shape scores to be the mean normalized relative distance between gesture and template points
        shape_scores.append(distances / 100)
    return shape_scores


# get value for alpha (to be used for calculating location scores)
# alpha is a linear function; number of sample points is 100
alpha = np.empty(100)
# give lowest weight to middle point, increase rest of points’ weight linearly towards the two ends
for i in range(50):
    a = i/2450
    alpha[50 - i - 1], alpha[50 + i] = a, a


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.
    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :return:
        A list of location scores.
    '''
    radius = 15
    # TODO: Calculate location scores (12 points)
    # initialize location_scores and gesture_sample_points
    location_scores = np.empty(len(valid_template_sample_points_X))
    gesture_sample_points_XY = []
    for i in range(len(gesture_sample_points_X)):
        gesture_sample_points_XY.append([gesture_sample_points_X[i], gesture_sample_points_Y[i]])
    # determine distance between each gesture and corresponding template point
    for i in range(len(valid_template_sample_points_X)):
        template_sample_points_XY = []
        for j in range(100):
            template_sample_points_XY.append([valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]])
        dist = distance_matrix(gesture_sample_points_XY, template_sample_points_XY)
        # get min for gesture and template corresponding points
        template_min_dist, gesture_min_dist = np.amin(dist[0]), np.amin(dist[1])
        # if a gesture or template point is outside of radius channel, get alpha * delta value for each point
        if np.any(gesture_min_dist > radius) or np.any(template_min_dist > radius):
            delta = np.diag(dist)
            location_scores[i] = np.sum(alpha * delta)
    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.7
    # TODO: Set your own location weight
    location_coef = 1 - shape_coef
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.
    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    # determine based word (using formula given in slides)
    final_scores = list(integration_scores[i] * (1 - probabilities[valid_words[i]]) for i in range(len(integration_scores)))
    sorted(final_scores)
    min_score = min(final_scores)
    w = list(valid_words[i] for i in range(len(final_scores)) if final_scores[i] == min_score)
    min_words = min(len(w), n)
    return ' '.join(w[:min_words])


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    start_time = time.time()
    data = json.loads(request.get_data())
    
    print(data)

    gesture_points_X, gesture_points_Y = [], []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)
    # normalize gesture points (only done once; used in pruning and shape scores)
    norm_gesture_sample_points_X, norm_gesture_sample_points_Y = normalize(gesture_sample_points_X, gesture_sample_points_Y)
    # do pruning
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(norm_gesture_sample_points_X,
        norm_gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y)

    best_word = "No best word"
    if len(valid_words) != 0:
        # get shape scores with normalized points
        shape_scores = get_shape_scores(norm_gesture_sample_points_X, norm_gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)
        # get location scores with un-normalized gesture points
        location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)
        # get integration scores
        integration_scores = get_integration_scores(shape_scores, location_scores)
        # get best word
        best_word = get_best_word(valid_words, integration_scores)
    end_time = time.time()
    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
