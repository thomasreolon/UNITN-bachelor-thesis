import urllib.request as ur
import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import torch
from torchvision import transforms
from torchvision import models
import io
import json
from PIL import Image

model_path = '../models/image_torch_res50.pth'
labels = '../models/image_torch_classes.json'

if __name__=='__main__':
    model_path = '../../'+model_path
    labels = '../../'+labels


def load_my_categories():
    with open(labels, 'r') as fin:
        mycat = json.load(fin)
    final = {}
    for k, v in mycat.items():  # 397 animals
        for num in v:
            final[num] = k
    def update_interests(inter, tensor):
        for i, perc in enumerate(tensor):
            if perc > 0.43 and i in final:
                a = final[i]
                inter[a] += 1
            elif perc > 0.43 and i<397:
                inter['animals'] += 1
    def get_empty_interests():
        return {k: 0 for k in mycat}

    return update_interests, get_empty_interests


def init(client):
    client.model = models.wide_resnet50_2(pretrained=False)
    checkpoint = torch.load(model_path)
    client.model.load_state_dict(checkpoint)
    client.model.eval()
    client.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    client.update_interests, client.new_interests_dict = load_my_categories()

def url_to_image(url):
    req = ur.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = ur.urlopen(req)
    except Exception:
        return None
    image_file = io.BytesIO(resp.read())
    return Image.open(image_file)


def main(name='', id='0'):
    mqtt.MQTTClient(process_type=name, funct=process_image, init=init, id=id, type=mqtt.CLASSIFIER)



def process_image(client:mqtt.MQTTClient, user:ds.User):
    if user.img_topics is not None:
        raise ds.UnableToProcessUser('had already a value')
    images = []
    # get images
    for act in user.activities:
        if act.images is not None:
            images += act.images

    img_topics = client.new_interests_dict()
    images = images[:20]
    for url in images:
        try:
            input_image = url_to_image(url)
            if input_image:
                input_tensor = client.transform(input_image)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = client.model(input_batch)
                output = torch.nn.functional.softmax(output[0], dim=0)

                client.update_interests(img_topics, output)
            for k,v in img_topics.items():
                img_topics[k] = v/(len(images)+2) # standardize
        except : pass
    return ds.User(img_topics=img_topics), f'{img_topics} | https://twitter.com/{user.username}'



if __name__ == '__main__':
    cli = lambda :None
    init(cli)
    process_image(cli, ds.User(activities=[{'images':['https://www.accademiadelprofumo.it/wp-content/uploads/2017/01/loto.jpg','https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559__340.jpg','https://cdn.cronachemaceratesi.it/wp-content/uploads/2019/11/matrimonio.jpg']}]))



