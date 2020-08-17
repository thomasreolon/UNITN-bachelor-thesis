import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
import pickle

path_to_model = '../models/mbti_classifier.st'
traits = ['T','S','E','J']

if __name__=='__main__':
    path_to_model = '../../' + path_to_model
####################### FUNCTIONS #############################################

def clean_post(post: str, hard=True):

    # manually clean the post
    # lower and apices
    post = post.lower()
    post = re.sub('\'|"', '', post)

    # users, hashtag and urls
    post = re.sub('@\w+', '@user', post)
    post = re.sub('#', '', post)
    post = re.sub('htt(p|ps)://\w+(\.\w+|/\w+)+', '@url ', post)
    # words with numbers in them
    if hard:
        post = re.sub('(\d*[a-z]+\d+[a-z]*\d*)+', '', post)
    # separators, non standards letters
    post = re.sub('\.|,|;|-|_', ' ', post)
    post = re.sub('[^ abcdefghilmnopqrstuvzwyxjk<>=\(\)$€]', '', post)

    return post

def process_data(client:mqtt.MQTTClient, user:ds.User):
    if user.mbti is not None:
        raise ds.UnableToProcessUser('already completed')
    posts = []
    for post in user.activities:
        if 'text' in post:
            posts.append(post.text)

    corpus = ' . '.join(posts)

    X = client.mbti_classifier['vect'].transform([corpus]).toarray()
    mbti = {}
    for f in traits:
        mbti[f] = int(client.mbti_classifier[f].predict(X)[0])
    return ds.User(mbti=mbti), f'{mbti} | https://twitter.com/{user.username}'



def init(client):
    st = open(path_to_model, 'rb').read()
    client.mbti_classifier = pickle.loads(st)
    # client.mbti_classifier = joblib.load(path_to_model)


def main(process_type='mbti_classifier', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_data, init=init,id=id, type=mqtt.CLASSIFIER)

if __name__ == '__main__':
    main()