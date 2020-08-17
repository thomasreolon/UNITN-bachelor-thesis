import cv2
import dlib
import numpy as np
import urllib.request as ur
from keras.applications import ResNet50
from keras.layers import Dense
from keras.models import Model
import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds

weight_file = "../models/age_weights.hdf5"
margin = 20

def url_to_image(url):
    resp = ur.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def get_age(image_path, client):
    img = url_to_image(image_path)
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = client.detector(input_img, 1)
    faces = np.empty((len(detected), client.img_size, client.img_size, 3))

    predicted_ages = None
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (client.img_size, client.img_size))

        # predict ages and genders of the detected faces
        results = client.model.predict(faces)
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results.dot(ages).flatten()
    return predicted_ages


def init(client:mqtt.MQTTClient):
    # for face detection
    client.detector = dlib.get_frontal_face_detector()

    # load model and weights
    base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3), pooling="avg")
    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)
    model = Model(inputs=base_model.input, outputs=prediction)
    model.load_weights(weight_file)
    client.img_size = model.input.shape.as_list()[1]
    client.model = model

def process(client:mqtt.MQTTClient, user:ds.User):
    image_path = user.photo
    ages = get_age(image_path, client)

    if (ages is None or len(ages)>1) and user.bg_photo:
        # 0 or >1 faces in te profile picture
        ages2 = get_age(user.bg_photo, client)
        if ages2 is None and ages is None:   # never got a result
            raise ds.UnableToProcessUser(f'not able to find an age in both img for:{user.id}')
        elif ages is None: # first age in the 2 photo (ages2[0])
            age = ages2[0]
        elif ages2 is None:# first age in the first photo (ages[0])
            age = ages[0]
        else: # the averages between the most similar ages in both photos
            min_dis, age = 100000, 0
            if ages2 is not None:
                for a1 in ages:
                    for a2 in ages2:
                        if (a1-a2)**2 < min_dis:
                            min_dis = (a1-a2)**2
                            age = (a1+a2)/2
    elif ages is not None:
        # only one face in the profile picture
        age = ages[0]
    else:
        raise ds.UnableToProcessUser(f'not able to find an age for:{user.id}')

    # draw results
    return ds.User(age=age), f"found an age for:{user.id}"



def main(process_type='mbti_classifier', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process, init=init, type=mqtt.CLASSIFIER)

if __name__=='__main__':
    main()