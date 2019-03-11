import string
from operator import itemgetter
import re
import os, csv
from numpy import unique
import enchant
import requests
from bs4 import BeautifulSoup as bsoup
import json
import numpy
from sklearn.preprocessing import LabelEncoder
import pickle
from run import DATADIR


class HUFFMAN:

    def __init__(self):
        self.codemap = {}
        self.tree = None

    def frequency(self, str):
        freqs = {}
        for ch in str:
            freqs[ch] = freqs.get(ch, 0) + 1
        return freqs

    def sortFreq(self, freqs):
        letters = freqs.keys()
        tuples = []
        for let in letters :
            tuples.append((freqs[let],let))
        tuples.sort()
        return tuples

    def buildTree(self, tuples):
        while len(tuples) > 1 :
            leastTwo = tuple(tuples[0:2])                  
            theRest  = tuples[2:]                         
            combFreq = leastTwo[0][0] + leastTwo[1][0]
            tuples   = theRest + [(combFreq,leastTwo)]
            sorted(tuples, key=itemgetter(0))           
        return tuples[0]           

    def trimTree(self, tree):
        p = tree[1]                 
        if type(p) == type(""):
             return p             
        else:
             return (self.trimTree(p[0]), self.trimTree(p[1])) 

    def assignCodes(self, node, pattrn=''):
        if type(node) == type("") :
            self.codemap[node] = pattrn            
        else:
            self.assignCodes(node[0], pattrn+"0")   
            self.assignCodes(node[1], pattrn+"1")

    @property
    def encoder_map(self):
        assert (len(self.codemap.keys())>0), "No Code Map Found"
        return self.codemap

    @property
    def decoder_map(self):
        assert (len(self.codemap.keys()) > 0), "No Code Map Found"
        return dict([(v, k) for k, v in self.codemap.items()])

    def encode_string(self, str):
        freqs = self.frequency(str)
        tuples = self.sortFreq(freqs)
        self.tree = self.buildTree(tuples)
        self.tree = self.trimTree(self.tree)
        self.assignCodes(self.tree)
        output = ""
        for ch in str:
            output += self.codemap[ch]
        return output

    def decode_string(self, binary):
        output = ""
        p = self.tree
        for bit in binary:
            if bit == '0':
                p = p[0] 
            else:
                p = p[1]
            if type(p) == type(""):
                output += p
                p = self.tree
        return output

    def initialize(self, fillvalue):
        self.encode_string("0123456789,:;'abcdefghijklmnopqrstuvwxyz+_~-=&$#" + " " + '"' + "\/" + "/")
        return {"huffman-encoder":self.encoder_map, "huffman-decoder": self.decoder_map, "fillvalue": fillvalue}


class SPELLCHECK:

    def __init__(self, dct, **kwargs):
        if dct == "custom":
            try:
                w = kwargs.get("words_list")
            except KeyError as err:
                raise KeyError("Must specify 'words_list' used to populate dict")
            else:
                self.dct = enchant.pypwl.PyPWL()
                self.BuildCustomDict(words=w)
        elif dct.find("_") > -1 or dct is None:
            self.dct = enchant.Dict(dct)

    def BuildCustomDict(self, words):
        for word in words:
            self.dct.add_to_pwl(word=word)

    def ValidWord(self, word):
        return self.dct.check(word=word)

    def Suggested(self, string, autoselect=0):
        try:
            sugg = self.dct.suggest(string)
        except Exception as err:
            print(err)
        else:
            if len(sugg) == 0:
                return None
            else:
                return sugg[autoselect]


class config:

    def __init__(self, *args, **kwargs):
        self.file = kwargs.get("file", "/".join([os.getcwd(), "config.json"]))

    @property
    def load(self):
        with open(self.file, mode="r") as f:
            js = json.load(f)
            return js

    def save(self, cfg_obj):
        with open(self.file, mode="w") as f:
            json.dump(cfg_obj, f)
            f.flush()
        print("config file updated")
        f.close()

    @property
    def WALMART(self):
        return self.load['WALMART']

    @property
    def USER(self):
        js = self.load["USER"]
        return js

    def _check_section(self, sect):
        if sect in ["USER", "GLOBAL"]:
            return True
        else:
            return False

    def _check_option(self, sect, o):
        if self._check_section(sect=sect) is True:
            assert (o in [k for k in self.load[sect].keys()] is True), "Invalid option %s " % o
            return True
        else:
            raise KeyError("Invalid Section %s" % sect)

    def update(self, section, key, value, option=None):
        if option is None:
            if self._check_section(sect=section) is True:
                self.load[section].update({key: value})
        elif option is not None:
            if self._check_section(sect=section) is True:
                if self._check_option(sect=section, o=option) is True:
                    self.load[section][option].update({key: value})
                else:
                    raise KeyError("Invalid Option")
            else:
                raise KeyError("Invalid Section")
        else:
            print("Missing Required Values")


class ItemCollections:

    @staticmethod
    def _retrieve_dist(items):
        uniques = unique(items, return_counts=True)
        lst = list(map(lambda x, y: {"abbr": x[:4], "name": x, "count": y, "pct": y / sum(uniques[1])},
                       uniques[0], uniques[1]))
        m = []
        for l in range(len(lst)):
            lst[l].update({"id": l})
            m.append(lst[l])
        return m

    @staticmethod
    def _get(type):
        _file = "/".join([DATADIR, "%s.ls" % type])
        if os.path.isfile(_file):
            with open(_file, mode="r") as f:
                readr = csv.DictReader(f, fieldnames=["id", "abbr", "name", "count", "pct"])
                next(readr)
                rows = [row for row in readr]
                return rows
        else:
            with open("/".join([DATADIR, 'receipt_categories', "labeled-updated.csv"]), mode="r") as f:
                readr = csv.DictReader(f, fieldnames=["id", "product_text", "retailer", "category"])
                next(readr)
                rows = ItemCollections._retrieve_dist(items=[row["%s" % type] for row in readr])
                with open(_file, mode="w") as fx:
                    writr = csv.DictWriter(fx, fieldnames=["id", "abbr", "name", "count", "pct"])
                    writr.writeheader()
                    writr.writerows(rows)
                fx.close()
            f.close()
            return rows

    @property
    def Categories(self):
        return ItemCollections._get(type="category")

    @property
    def Retailers(self):
        return ItemCollections._get(type="retailer")

    @property
    def SubstitutionStrings(self):
        arr = ["kkroger", "kroger", "rogr", "krger", "kroeer", "kkroer",
               "roger", "kkrog", "kkroger", "krgr", "kkro", "krog", "koer", "k0er",
               "kkr", "kro", "kr0", "kr0g", "kr0ger", "kkr0", "kkr0ger",
               "koe", "r0ger", "safeway", "safewy", "safw", "sfway", "sfwy",
               "ppublix", "publix", "publx", "publ", "kroge", "ppc", "subst",
               "sub", "pcs", "pc", "wt"]
        return arr


class EncodingMaps:

    def __init__(self, type=None):
        super().__init__()
        cats = ItemCollections().Categories
        self.categories = {
            "encode-map": dict(map(lambda x: (cats[x]["name"], int(cats[x]["id"])),
                                   range(0, len(cats)))),
            "decode-map": dict(map(lambda x: (int(cats[x]["id"]), cats[x]["name"]),
                                   range(0, len(cats))))
        }

        self.features = self.setup_features_encoder(type=type, fill_value=-99)

    def setup_features_encoder(self, type, fill_value):
        if type == "huffman":
            return HUFFMAN().initialize(fillvalue=fill_value)
        elif type == "LabelEncoder":
            lab = LabelEncoder()
            lab.fit(y=list(string.ascii_letters+string.digits+string.punctuation))
            return {"label-encoder": lambda x: lab.transform(x),
                    "label-decoder":lambda x: lab.inverse_transform(x),
                    "fillvalue": fill_value}

    def encodedTextGenerator(self, string_lst, **kwargs):
        encoder = kwargs.get("encoder", "huffman")
        if encoder == "huffman":
            for string in string_lst:
                try:
                    x = str(string)
                except TypeError:
                    raise TypeError("strings in string_array must be unicode friendly")
                else:
                    yield list(map(lambda i: self.features["%s-encoder" % encoder][i], list(x)))

    def _remove_spacers(self, lst):
        if self.features["huffman-filler"] in lst:
            lst.remove(self.features["huffman-filler"])
            self._remove_spacers(lst)
        else:
            pass
        return lst

    def decodedTextGenerator(self, encoded_lst, **kwargs):

        encoder = kwargs.get("encoder", "huffman")

        assert (type(encoded_lst) in [list, tuple, numpy.ndarray, set]), "encoded_lst parameter must be type list or other iterable not type %s. DUH! " % type(encoded_lst)
        if encoder == "huffman":
            l = "".join(list(map(lambda x: self.features["huffman-decoder"][x], self._remove_spacers(lst=encoded_lst))))
            yield l

    def categoriesEncoder(self, categories_lst):
        assert (type(categories_lst) in [list, tuple, set, numpy.ndarray]), "string_lst parameter must be type list or other iterable not type %s. DUH! " % type(categories_lst)
        return [self.categories["encode-map"][x] for x in categories_lst]

    def categoriesDecoder(self, categories_lst):
        assert (type(categories_lst) in [list, tuple, set, numpy.ndarray]), "categories_lst parameter must be type list or other iterable not type %s. DUH! " % type(categories_lst)
        return [self.categories["decode-map"][int(x)] for x in categories_lst]


def is_notanitem(string):
    NOTITEM_STRINGS = ["sav", "fuel", "pnts", "points", "pts", "you",
                       "parti", "part", "prtic", "par", "for", "chg",
                       "chng", "chan", "svng", "svn", "svin", "amo",
                       "amount", "amt", "ouv", "ount", "f0r", "avin",
                       "rew", "wards", "war", "rewrds", "car", "wrds", "ward",
                       "ord", "orde", "sains", "cas", "cah", "cash",
                       "back", "bak", "refn", "refun", "rfnd", "refund", "fund",
                       "ttl", "valu", "value", "discount", "empl", "employee",
                       "loyal", "loy", "disc", "tax", "perc", "percent", "pct",
                       "bala", "baac", "ance", "rnce", "lnce", "cred", "crd", "deb",
                       "debit", "credit", "purch", "chas", "expi", "point", "visit",
                       "regis", "rgist", "limi", 'lmt', "code", "sku", "$", "pric",
                       "pri", "prc", "gal", "gas", "check", "bonu", "debi", "card"]
    if string.isdigit() or any([string.find(x) for x in NOTITEM_STRINGS]) > -1:
        return True
    else:
        return False


def find_location():
    url = "https://www.iplocation.net/find-ip-address"
    req = requests.get(url)
    soup = bsoup(req.text, "html5lib")
    table = soup.find("table", {"class": "iptable"})
    body = table.find("tbody")
    rows = body.findAll("tr")
    city = None
    for row in rows:
        if row.find("th").string == "IP Location":
            loc = next(row.find("td").strings).split(", ")
            city = loc[0]

        else:
            pass
    return city


def product_text_cleaner(txt):
    kroger_pattern = re.compile('.?kr(.*?)\s')
    text = txt.lower()
    text = re.sub(kroger_pattern, "", text)
    coll = ItemCollections()
    for r in coll.SubstitutionStrings:
        if text.find(r) > -1:
            text = product_text_cleaner(txt=text.replace(r, ""))
    return text


def pickle_classifier(clf, file):
    with open(file, mode="wb") as f:
        print("Saving Fitted Classifier to File")
        pickle.dump(obj=clf, file=f)
    print("Saved.")
    f.close()


def unpickle_classifier(file):
    with open(file, mode="rb") as f:
        p = pickle.load(f)
        return p
