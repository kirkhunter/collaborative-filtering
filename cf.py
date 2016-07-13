import argparse
import numpy as np
import pandas as pd
import time


def parse_argument():
    parser = argparse.ArgumentParser(description='Parsing a file')
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--train', nargs=1, required=True)
    args = parser.parse_args()
    return args.train[0], args.test[0]


def store_ratings(train_data_file):
    ratings = {}
    movies = {}
    with open(train_data_file, 'rb') as f:
        for line in f:
            movie, user, rating = line.split(',')
            movie, user, rating = int(movie), int(user), float(rating)
            if user not in ratings:
                ratings[user] = {movie:rating}
                ratings[user]['movies'] = set([movie])
                ratings[user]['total'] = rating
                ratings[user]['count'] = 1
            else:
                ratings[user][movie] = rating
                ratings[user]['movies'].add(movie)
                ratings[user]['total'] += rating
                ratings[user]['count'] += 1
            if movie not in movies:
                movies[movie] = {user:rating}
                movies[movie]['users'] = set([user])
                movies[movie]['total'] = rating
                movies[movie]['count'] = 1
            else:
                movies[movie][user] = rating
                movies[movie]['users'].add(user)
                movies[movie]['total'] += rating
                movies[movie]['count'] += 1 
    return ratings, movies


def create_movie_averages_dict(movies_dict, movies_set):
    movie_averages_dict = {}
    total = 0
    count = 0
    for movie in movies_set:
        avg = movies_dict[movie]['total'] / float(movies_dict[movie]['count'])
        movie_averages_dict[movie] = avg
        total += avg
        count += 1 
    movie_averages_dict['all_movies'] = total / float(count)
    return movie_averages_dict


def create_user_averages_dict(users_dict, users_set):
    user_averages_dict = {}
    total = 0
    count = 0
    for user in users_set:
        avg = users_dict[user]['total'] / float(users_dict[user]['count'])
        user_averages_dict[user] = avg
        total += avg
        count += 1 
    user_averages_dict['all_users'] = total / float(count)
    return user_averages_dict
    

def create_similarities_dict(users_set, users_dict, movies_dict, 
                             user_averages_dict):    
    similarities_dict = {}
    for user1 in users_set:
        if user1 not in similarities_dict:
            similarities_dict[user1] = {user1:0}
        for user2 in users_set:
            if user1 < user2:
                sim = similarity(user1, user2, users_dict, movies_dict, user_averages_dict)
                similarities_dict[user1][user2] = sim
    return similarities_dict


def get_common_movies(user_i, user_j, users_dict):
    try:
        i_movies = users_dict[user_i]['movies']
        j_movies = users_dict[user_j]['movies']
    except KeyError:
        common_movies = []
    else:
        common_movies = i_movies.intersection(j_movies)
    return common_movies


def similarity(user_i, user_j, users_dict, movies_dict, user_averages_dict):
    common_movies = get_common_movies(user_i, user_j, users_dict)
    if common_movies:
        avg_rating_i = user_averages_dict[user_i]
        avg_rating_j = user_averages_dict[user_j]
        numerator = 0
        diff_1 = 0
        diff_2 = 0
        for movie in common_movies:
            i_rating = users_dict[user_i][movie]
            j_rating = users_dict[user_j][movie]
            numerator += (i_rating - avg_rating_i) * (j_rating - avg_rating_j)
            diff_1 += (i_rating - avg_rating_i)**2
            diff_2 += (j_rating - avg_rating_j)**2
        denominator = np.sqrt(diff_1 * diff_2)
        return numerator / denominator if denominator else 0
    return 0


def get_user_average(user, user_averages_dict):
    is_new_user = False
    try:
        user_i_avg_rating = user_averages_dict[user]
    except KeyError:
        user_i_avg_rating = user_averages_dict['all_users']
        is_new_user = True
    return user_i_avg_rating, is_new_user


def get_movie_average(movie, movie_averages_dict):
    is_new_movie = False
    try:
        movie_avg_rating = movie_averages_dict[movie]
    except KeyError:
        movie_avg_rating = movie_averages_dict['all_movies']
        is_new_movie = True
    return movie_avg_rating, is_new_movie


def get_similarity(u_i, u_j, sim_dict):
    try:
        sim = sim_dict[u_i][u_j] if u_i < u_j else sim_dict[u_j][u_i]
    except KeyError:
        sim = 0
    return sim

###############################################################################################
def get_similarity(u_i, u_j, sim_dict):
    try:
        sim = sim_dict[u_i][u_j] if u_i < u_j else sim_dict[u_j][u_i]
    except KeyError:
        sim = similarity(u_i, u_j, users_dict, movies_dict, user_averages_dict)
    return sim
###############################################################################################

def get_movie_users(movie, movies_dict):
    try:
        movie_users = movies_dict[movie]['users']
    except KeyError:
        movie_users = []
    return movie_users
        

def make_prediction(user_i, movie_k, sim_dict, users_dict, 
                    user_averages_dict, movie_averages_dict, movies_dict):
    
    weight = 0
    val = 0
    user_i_avg_rating, is_new_user = get_user_average(user_i, user_averages_dict)
    movie_k_avg_rating, is_new_movie = get_movie_average(movie_k, movie_averages_dict)
    movie_k_users = get_movie_users(movie_k, movies_dict)
    if is_new_movie:
        return user_i_avg_rating
    elif is_new_user:
        return movie_k_avg_rating 
    for u_j in movie_k_users:
        sim = get_similarity(user_i, u_j, sim_dict)
        weight += abs(sim)
        try:
            val += sim * (users_dict[u_j][movie_k] - user_averages_dict[u_j])
        except KeyError:
            pass
    inv_weight = 1 / weight if weight else 0
    pred = user_i_avg_rating + inv_weight * val
    if pred < 1: return 1.0
    elif pred > 5: return 5.0
    return pred


def write_predictions_to_output(test_frame, users_dict, similarities_dict, 
                                user_averages_dict, movie_averages_dict, movies_dict):

    movies_list = list(test_frame['movie_id'])
    users_list = list(test_frame['customer_id'])
    test_observations = zip(users_list, movies_list)
    predictions = map(lambda (u, m): make_prediction(u, m, similarities_dict, users_dict, 
                                                     user_averages_dict, movie_averages_dict, 
                                                     movies_dict), 
                                                        test_observations)

    test_frame['prediction'] = predictions
    test_frame['error'] = test_frame['rating'] - test_frame['prediction']

    mean_abs_error = np.mean(abs(test_frame['error']))
    rmse = np.sqrt(np.mean(test_frame['error']**2))

    out = test_frame[['movie_id', 'customer_id', 'rating', 'prediction']]

    out.to_csv('predictions0.txt', index=False)

    print ""
    print "Mean Absolute Error: %.5f" % mean_abs_error
    print "Root Mean Squared Error: %.5f" % rmse
    print ""
    print "predictions.txt has been saved to file"


if __name__ == '__main__':

    start = time.time()

    train_file, test_file = parse_argument()

    df_test = pd.read_csv(test_file, sep=',', header=None)

    df_test.columns = ['movie_id', 'customer_id', 'rating']

    users, movies = store_ratings(train_file)

    users_set = [user for user in users.keys() if isinstance(user, int)]
    
    movies_set = [movie for movie in movies.keys() if isinstance(movie, int)]

    user_averages = create_user_averages_dict(users, users_set)

    movie_averages = create_movie_averages_dict(movies, movies_set)

    similarities = create_similarities_dict(users_set, users, 
                                            movies, user_averages)

    write_predictions_to_output(df_test, users, similarities, 
                                    user_averages, movie_averages, movies)

    print "Runtime: %.5f seconds" % (time.time()-start)
    print ""

