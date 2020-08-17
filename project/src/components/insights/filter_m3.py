import mycommons.mqtt_wrap as mqtt
import mycommons.data_structure as ds
import pycld2 as cld2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
from torchvision import transforms
import torchvision
import urllib.request as ur
import torch
import unicodedata
import io
import gc
import re
import pickle
import resource
from PIL import Image
import pymongo

USE_IMG=False

model_fold = '../models'
TRESHOLD_ORG = 0.55
supported_langs = {'en', 'cs', 'fr', 'nl', 'ar', 'ro', 'bs', 'da', 'it', 'pt', 'no', 'es', 'hr', 'tr', 'de', 'fi', 'el', 'he', 'ru', 'bg', 'hu', 'sk', 'et', 'pl', 'lv', 'sl', 'lt', 'ga', 'eu', 'mt', 'cy', 'rm', 'is', 'un'}
####################### FUNCTIONS #############################################

transf_img = transforms.ToTensor()

if __name__=='__main__':
    model_fold = '../../' + model_fold
def init(client):
    client.m3 = get_model()
    client.m3text = get_model(full=False)
    torch.no_grad()
    client.users = pymongo.MongoClient('localhost', 27017)['personas']['users']

def guess_language(user:ds.User):
    try:
        doc = [user.description]
        for i, act in enumerate(user.activities):
            if i == 5:
                break
            elif act.text:
                doc.append(act.text)
        doc = " . ".join(doc)
        lang = cld2.detect(''.join([i for i in doc if i.isprintable()]), bestEffort=True)[2][0][1]
        res = UNKNOWN_LANG if lang not in LANGS else lang
    except Exception:
        res = user.language

    if res not in supported_langs:
        res = 'un'
    return res


# model parameter
BATCH_SIZE = 128
EMBEDDING_OUTPUT_SIZE = 128
EMBEDDING_INPUT_SIZE = 3035
EMBEDDING_OUTPUT_SIZE_ASCII = 16
EMBEDDING_INPUT_SIZE_ASCII = 128
LSTM_LAYER = 2
LSTM_LAYER_DES = 2
LSTM_HIDDEN_SIZE = 256
LSTM_OUTPUT_SIZE = 128
LINEAR_OUTPUT_SIZE = 2
VISION_OUTPUT_SIZE = 2048
USERNAME_LEN = 30
SCREENNAME_LEN = 16
DES_LEN = 200

# model dump parameter
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'full_model': ['https://nlp.stanford.edu/~zijwang/m3inference/full_model.mdl',
                   'https://blablablab.si.umich.edu/projects/m3/models/full_model.mdl'],
    'text_model': ['https://nlp.stanford.edu/~zijwang/m3inference/text_model.mdl',
                   'https://blablablab.si.umich.edu/projects/m3/models/text_model.mdl']
}

PRETRAINED_MODEL_MD5_MAP = {
    'full_model': '7dd11b9d89d7fd209e3baa0058baa4a1',
    'text_model': 'c9a9fbd953b3ad5d84e792c3c50392ad'
}

# unicode parameter
UNICODE_CATS = 'Cc,Zs,Po,Sc,Ps,Pe,Sm,Pd,Nd,Lu,Sk,Pc,Ll,So,Lo,Pi,Cf,No,Pf,Lt,Lm,Mn,Cn,Me,Mc,Nl,Zl,Zp,Cs,Co'.split(",")

# language parameter
LANGS = ['en', 'cs', 'fr', 'nl', 'ar', 'ro', 'bs', 'da', 'it', 'pt', 'no', 'es', 'hr', 'tr', 'de', 'fi', 'el', 'he',
         'ru', 'bg', 'hu', 'sk', 'et', 'pl', 'lv', 'sl', 'lt', 'ga', 'eu', 'mt', 'cy', 'rm', 'is', 'un']
LANGS = {k: v for v, k in enumerate(LANGS)}
UNKNOWN_LANG = 'un'

# embedding parameter
EMBEDDING_INPUT_SIZE_LANGS = len(LANGS) + 1
EMBEDDING_OUTPUT_SIZE_LANGS = 8
EMB = pickle.load(open(model_fold+"/emb.pkl", "rb"))

PRED_CATS = {
    'gender': ['male', 'female'],
    'age': ['<=18', '19-29', '30-39', '>=40'],
    'org': ['non-org', 'is-org']
}

def get_model(full=True):
    if full:
        model = M3InferenceModel()
        model.load_state_dict(torch.load(model_fold+'/full_model.mdl'))
        model.eval()
        return model
    else:
        model = M3InferenceTextModel()
        model.load_state_dict(torch.load(model_fold + '/text_model.mdl'))
        model.eval()
        return model


def pack_wrapper(sents, lengths):
    lengths_sorted, idx_sorted = lengths.sort(descending=True)
    sents_sorted = sents[idx_sorted]
    packed = pack_padded_sequence(sents_sorted, lengths_sorted, batch_first=True)
    return packed, idx_sorted


def unpack_wrapper(sents, idx_unsort):
    h, _ = pad_packed_sequence(sents, batch_first=True)
    h = torch.zeros_like(h).scatter_(0, idx_unsort.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)
    return h


def normalize_url(sent):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '\u20CC', sent)


def normalize_space(sent):
    return sent.replace("\t", " ").replace("\n", " ").replace("\r", " ")



def image_loader(image_name):
    if image_name and 'http' in image_name:
        req = ur.Request(image_name, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            resp = ur.urlopen(req)
        except Exception:
            return None
        with io.BytesIO(resp.read()) as image_file:
            image = Image.open(image_file)
            res = transf_img(image)
            del image
    else:
        return None
    return res


def process_user(client:mqtt.MQTTClient, user:ds.User):
    _id = user.id
    lang = guess_language(user)
    user.language = {"it": 0, "en": 0, "fr": 0, "de": 0, "sp": 0, "un":0}
    if lang in user.language.keys():
        user.language[lang] = 1
    else:
        user.language["un"] = 1

    username = normalize_space(user.name)
    screenname = normalize_url(normalize_space(user.username))
    des = normalize_space(user.description)
    img_path = user.photo if USE_IMG else 'no'

    image = image_loader(img_path)

    ############## preprocess - toTensor ################

    lang_tensor = LANGS[lang]

    username_tensor = [0] * USERNAME_LEN
    if username.strip(" ") == "":
        username_tensor[0] = EMB["<empty>"]
        username_len = 1
    else:
        if len(username) > USERNAME_LEN:
            username = username[:USERNAME_LEN]
        username_len = len(username)
        username_tensor[:username_len] = [EMB.get(i, len(EMB) + 1) for i in username]

    screenname_tensor = [0] * SCREENNAME_LEN
    if screenname.strip(" ") == "":
        screenname_tensor[0] = 32
        screenname_len = 1
    else:
        if len(screenname) > SCREENNAME_LEN:
            screenname = screenname[:SCREENNAME_LEN]
        screenname_len = len(screenname)
        screenname_tensor[:screenname_len] = [ord(i) for i in screenname]

    des_tensor = [0] * DES_LEN
    if des.strip(" ") == "":
        des_tensor[0] = EMB["<empty>"]
        des_len = 1
    else:
        if len(des) > DES_LEN:
            des = des[:DES_LEN]
        des_len = len(des)
        des_tensor[:des_len] = [EMB.get(i, EMB[unicodedata.category(i)]) for i in des]

    y_pred = []
    with torch.no_grad():
        if image is not None:
            result = 'IMAGE | '
            data_batch = [lang_tensor, torch.LongTensor(username_tensor), username_len, torch.LongTensor(
                screenname_tensor), screenname_len, torch.LongTensor(des_tensor), des_len, image]
            data_batch = [torch.as_tensor(x).unsqueeze(0) for x in data_batch]
            pred = client.m3(data_batch)
            y_pred.append([_pred.detach().cpu().numpy() for _pred in pred])
        else:
            result = 'TEXT | '
            data_batch = [lang_tensor, torch.LongTensor(username_tensor), username_len, torch.LongTensor(
                screenname_tensor), screenname_len, torch.LongTensor(des_tensor), des_len]
            data_batch = [torch.as_tensor(x).unsqueeze(0) for x in data_batch]
            pred = client.m3text(data_batch)
            y_pred.append([_pred.detach().cpu().numpy() for _pred in pred])

    #return y_pred, 's'
    is_male = y_pred[0][0][0][0]
    ages = y_pred[0][1][0]
    non_org = y_pred[0][2][0][0]


    mem_usage='_____________________USAGE_________________________\n'+str(resource.getrusage(resource.RUSAGE_SELF))+'\n^^^^^^^^^^'
    if 'logger' in client.__dict__:
        client.logger.debug(mem_usage)
    else:
        print(mem_usage)

    if non_org>TRESHOLD_ORG:
        ## gender
        user.gender = float(is_male)

        ## age
        tmp = 0
        for i, k in enumerate(ages):
            tmp += (50/(1+2.71**(-i+2))+10)*k
        user.age = int(tmp)
        result += f'age={user.age}, isMale={user.gender} | https://twitter.com/{user.username}'
    else:
        result += f'seems to be a company  https://twitter.com/{user.username}'
        client.users.delete_one({'_id':user.id})
        user = None

    for in_f in data_batch:
        del in_f
    for ou_f in pred:
        del ou_f
    del data_batch
    gc.collect()

    return user, result



def main(process_type='filter_m3', id='0'):
    mqtt.MQTTClient(process_type=process_type, funct=process_user, init=init,id=id, type=mqtt.FILTER)


###################### Models ##########################


class M3InferenceModel(nn.Module):
    def __init__(self, device='cpu'):
        super(M3InferenceModel, self).__init__()

        self.device = device
        self.batch_size = -1

        self._init_vision_model()

        self.username_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                padding_idx=EMB['<empty>'])
        self.screenname_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                  padding_idx=EMB['<empty>'])
        self.des_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                           padding_idx=EMB['<empty>'])

        self.username_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.username_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                        out_features=EMBEDDING_OUTPUT_SIZE)
        self.username_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                     num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.username_dense)

        self.screenname_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_ASCII, EMBEDDING_OUTPUT_SIZE_ASCII,
                                             padding_idx=EMB['<empty>'])
        self.screenname_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE_ASCII + EMBEDDING_OUTPUT_SIZE_LANGS,
                                          out_features=EMBEDDING_OUTPUT_SIZE_ASCII)
        self.screenname_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE_ASCII, hidden_size=LSTM_HIDDEN_SIZE,
                                       num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.screenname_dense)

        self.des_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.des_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                   out_features=EMBEDDING_OUTPUT_SIZE)
        self.des_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                num_layers=LSTM_LAYER_DES, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.des_dense)

        merge_size = LSTM_HIDDEN_SIZE * 8

        self.merge_dense_co = nn.Linear(in_features=merge_size, out_features=LSTM_HIDDEN_SIZE)
        self.gender_out_dense_co = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.org_out_dense_co = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.age_out_dense_co = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=4)

        self._init_dense(self.merge_dense_co)
        self._init_dense(self.gender_out_dense_co)
        self._init_dense(self.org_out_dense_co)
        self._init_dense(self.age_out_dense_co)

    def _init_vision_model(self):
        self.vision_model = torchvision.models.densenet161(num_classes=LSTM_HIDDEN_SIZE * 2)

    def _init_dense(self, layer):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.uniform_(layer.bias)

    def _init_hidden(self):

        self.username_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)
        self.username_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)

        self.screenname_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)
        self.screenname_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)

        self.des_h0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE)
        self.des_c0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE)

    def forward(self, data, label=None):

        lang, username, username_len, screenname, screenname_len, des, des_len, fig = data
        self.batch_size = len(lang)
        self._init_hidden()

        username_lang_embed = self.username_lang_embed(lang)
        screenname_lang_embed = self.screenname_lang_embed(lang)
        des_lang_embed = self.des_lang_embed(lang)

        merge_layer = []

        username_embed = self.username_embed(username)

        username_embed = self.username_dense(torch.cat([username_embed,
                                                        username_lang_embed.unsqueeze(1).expand(self.batch_size,
                                                                                                USERNAME_LEN,
                                                                                                EMBEDDING_OUTPUT_SIZE_LANGS)],
                                                       2))

        username_pack, username_unsort = pack_wrapper(username_embed, username_len)
        self.username_lstm.flatten_parameters()
        username_out, (self.username_h0, self.username_c0) = self.username_lstm(username_pack, (
            self.username_h0, self.username_c0))
        username_output = unpack_wrapper(username_out, username_unsort)

        merge_layer.append(torch.cat([username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                      username_len - 1, :LSTM_HIDDEN_SIZE],
                                      username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                      torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        screenname_embed = self.screenname_embed(screenname)

        screenname_embed = self.screenname_dense(torch.cat([screenname_embed,
                                                            screenname_lang_embed.unsqueeze(1).expand(
                                                                self.batch_size, SCREENNAME_LEN,
                                                                EMBEDDING_OUTPUT_SIZE_LANGS)], 2))

        screenname_pack, screenname_unsort = pack_wrapper(screenname_embed, screenname_len)
        self.screenname_lstm.flatten_parameters()
        screenname_out, (self.screenname_h0, self.screenname_c0) = self.screenname_lstm(screenname_pack, (
            self.screenname_h0, self.screenname_c0))
        screenname_output = unpack_wrapper(screenname_out, screenname_unsort)

        merge_layer.append(torch.cat([screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                      screenname_len - 1, :LSTM_HIDDEN_SIZE],
                                      screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                      torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        des_embed = self.des_embed(des)

        des_embed = self.des_dense(torch.cat([des_embed,
                                              des_lang_embed.unsqueeze(1).expand(self.batch_size, DES_LEN,
                                                                                 EMBEDDING_OUTPUT_SIZE_LANGS)],
                                             2))

        des_pack, des_unsort = pack_wrapper(des_embed, des_len)
        self.des_lstm.flatten_parameters()
        des_out, (self.des_h0, self.des_c0) = self.des_lstm(des_pack, (self.des_h0, self.des_c0))
        des_output = unpack_wrapper(des_out, des_unsort)

        merge_layer.append(torch.cat(
            [des_output[torch.arange(0, self.batch_size, dtype=torch.int64), des_len - 1, :LSTM_HIDDEN_SIZE],
             des_output[torch.arange(0, self.batch_size, dtype=torch.int64),
             torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        vision_output = self.vision_model(fig) # CANNOT ALLOCATE MEMORY
        merge_layer.append(vision_output)
        merged_cat = torch.cat(merge_layer, 1)

        dense = F.relu(self.merge_dense_co(merged_cat), inplace=True)
        if label == "gender":
            return F.softmax(self.gender_out_dense_co(dense), dim=1)
        elif label == "age":
            return F.softmax(self.age_out_dense_co(dense), dim=1)
        elif label == "org":
            return F.softmax(self.org_out_dense_co(dense), dim=1)
        else:
            return F.softmax(self.gender_out_dense_co(dense), dim=1), \
                   F.softmax(self.age_out_dense_co(dense), dim=1), \
                   F.softmax(self.org_out_dense_co(dense), dim=1)



class M3InferenceTextModel(nn.Module):
    def __init__(self, device='cpu'):
        super(M3InferenceTextModel, self).__init__()
        self.device = device
        self.batch_size = -1

        self.username_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                padding_idx=EMB['<empty>'])
        self.screenname_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                  padding_idx=EMB['<empty>'])
        self.des_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                           padding_idx=EMB['<empty>'])

        self.username_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.username_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                        out_features=EMBEDDING_OUTPUT_SIZE)
        self.username_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                     num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.username_dense)

        self.screenname_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_ASCII, EMBEDDING_OUTPUT_SIZE_ASCII,
                                             padding_idx=EMB['<empty>'])
        self.screenname_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE_ASCII + EMBEDDING_OUTPUT_SIZE_LANGS,
                                          out_features=EMBEDDING_OUTPUT_SIZE_ASCII)
        self.screenname_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE_ASCII, hidden_size=LSTM_HIDDEN_SIZE,
                                       num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.screenname_dense)

        self.des_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.des_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                   out_features=EMBEDDING_OUTPUT_SIZE)
        self.des_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                num_layers=LSTM_LAYER_DES, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.des_dense)

        self.merge_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE * 6, out_features=LSTM_HIDDEN_SIZE)
        self.gender_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.org_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.age_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=4)

        self._init_dense(self.merge_dense)
        self._init_dense(self.gender_out_dense)
        self._init_dense(self.org_out_dense)
        self._init_dense(self.age_out_dense)

    def _init_dense(self, layer):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.uniform_(layer.bias)

    def _init_hidden(self):

        self.username_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)
        self.username_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)

        self.screenname_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)
        self.screenname_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE)

        self.des_h0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE)
        self.des_c0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE)

    def forward(self, data, label=None):

        lang, username, username_len, screenname, screenname_len, des, des_len = data
        self.batch_size = len(lang)
        self._init_hidden()

        username_lang_embed = self.username_lang_embed(lang)
        screenname_lang_embed = self.screenname_lang_embed(lang)
        des_lang_embed = self.des_lang_embed(lang)

        self.merge_layer = []

        username_embed = self.username_embed(username)

        username_embed = self.username_dense(torch.cat([username_embed,
                                                        username_lang_embed.unsqueeze(1).expand(self.batch_size,
                                                                                                USERNAME_LEN,
                                                                                                EMBEDDING_OUTPUT_SIZE_LANGS)],
                                                       2))
        username_pack, username_unsort = pack_wrapper(username_embed, username_len)
        self.username_lstm.flatten_parameters()
        username_out, (self.username_h0, self.username_c0) = self.username_lstm(username_pack, (
            self.username_h0, self.username_c0))
        username_output = unpack_wrapper(username_out, username_unsort)

        self.merge_layer.append(torch.cat([username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           username_len - 1, :LSTM_HIDDEN_SIZE],
                                           username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        screenname_embed = self.screenname_embed(screenname)

        screenname_embed = self.screenname_dense(torch.cat([screenname_embed,
                                                            screenname_lang_embed.unsqueeze(1).expand(
                                                                self.batch_size, SCREENNAME_LEN,
                                                                EMBEDDING_OUTPUT_SIZE_LANGS)], 2))

        screenname_pack, screenname_unsort = pack_wrapper(screenname_embed, screenname_len)
        self.screenname_lstm.flatten_parameters()
        screenname_out, (self.screenname_h0, self.screenname_c0) = self.screenname_lstm(screenname_pack, (
            self.screenname_h0, self.screenname_c0))
        screenname_output = unpack_wrapper(screenname_out, screenname_unsort)
        self.merge_layer.append(torch.cat([screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           screenname_len - 1, :LSTM_HIDDEN_SIZE],
                                           screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        des_embed = self.des_embed(des)

        des_embed = self.des_dense(torch.cat([des_embed,
                                              des_lang_embed.unsqueeze(1).expand(self.batch_size, DES_LEN,
                                                                                 EMBEDDING_OUTPUT_SIZE_LANGS)],
                                             2))

        des_pack, des_unsort = pack_wrapper(des_embed, des_len)
        self.des_lstm.flatten_parameters()
        des_out, (self.des_h0, self.des_c0) = self.des_lstm(des_pack, (self.des_h0, self.des_c0))
        des_output = unpack_wrapper(des_out, des_unsort)
        self.merge_layer.append(torch.cat(
            [des_output[torch.arange(0, self.batch_size, dtype=torch.int64), des_len - 1, :LSTM_HIDDEN_SIZE],
             des_output[torch.arange(0, self.batch_size, dtype=torch.int64),
             torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        merged_cat = torch.cat(self.merge_layer, 1)

        dense = F.relu(self.merge_dense(merged_cat), inplace=True)
        if label == "gender":
            return F.softmax(self.gender_out_dense(dense), dim=1)
        elif label == "age":
            return F.softmax(self.age_out_dense(dense), dim=1)
        elif label == "org":
            return F.softmax(self.org_out_dense(dense), dim=1)
        else:
            return F.softmax(self.gender_out_dense(dense), dim=1), \
                   F.softmax(self.age_out_dense(dense), dim=1), \
                   F.softmax(self.org_out_dense(dense), dim=1)







if __name__ == '__main__':
    # main()
    cli = lambda :None
    init(cli)
    for i in range(30):
        u, r = process_user(cli, ds.User(id=1323, language='it', name='Luca Tozzi', username='luca.tozzi', description='have a good day!', photo='xhttps://pbs.twimg.com/profile_images/2373443350/image.jpg'))
        print(u, '\n', r)




