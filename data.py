
import csv
import os
import random
import walmart
from numpy import unique

from utils import is_notanitem, ItemCollections, config, \
    HUFFMAN, DATADIR, product_text_cleaner

try:
    from io import StringIO as stringio
except ImportError:
    from StringIO import StringIO as stringio

huffman = HUFFMAN()
config = config()
categories = ItemCollections().Categories
retailers = ItemCollections().Retailers


class Reciepts:

    @property
    def labeled_products(self):
        return self.Items("labeled.csv")

    @property
    def unlabeled_products(self):
        return self.Items("unlabeled.csv")

    def Items(self, filename, fields=None):
        file = "/".join([DATADIR, "receipt_categories", filename])
        rows = []
        if fields is None:
            fileheader=["id", "product_text", "retailer", 'category']
        else:
            fileheader=fields
        with open(file, mode="r") as f:
            readr = csv.DictReader(f, fieldnames=fileheader)
            next(readr)
            for row in readr:
                yield row



class DataHandle(Reciepts):

    def __init__(self):
        super().__init__()
        self.labeled_items = [{"id": p['id'], "product_text": product_text_cleaner(p['product_text']), "retailer": p['retailer'].lower(), "category": p['category']}
                              for p in self.labeled_products]

        self.unlabeled_items = [{"id": p['id'], "product_text": product_text_cleaner(p['product_text']), "retailer": p['retailer'].lower(), "category": p['category']}
                                for p in self.unlabeled_products]
        self.request_count = 0

    def _list_weights(self, x):
        std = 1 / len(x)
        uni = [u for u in unique(x, return_counts=True)]
        wtd = dict(map(lambda x: (uni[0][x], uni[1][x]*std), range(len(uni[0]))))
        return wtd

    def contains_category(self, string):
        for x in categories:
            cat = x["name"]
            catparts = [cat[len(cat)-6-i: len(cat)-i] for i in range(len(cat)-6)]
            catparts.append(cat)
            fnd = list(filter(lambda x: string.find(x) > -1, catparts))
            if len(fnd) > 0:
                return x["name"]
            else:
                pass

    def category_wts(self, string):
        equal_weights = dict(map(lambda x: (x["name"], 1/len(categories)), [i for i in categories]))
        c = self.contains_category(string=string)
        if type(c) is str:
            return {c: 1.0}
        else:
            if is_notanitem(string) is True or len(string.replace(" ", "").strip()) <= 2:
                return {"not an item": 1.0}
            else:
                print("Running WalmartAPI Search...")
                try:

                    results = walmart.API().Search(qstring=string)
                except:
                    print("WalmartAPI Search Failed: Possibly Exceeded Daily Request Limit")
                    return equal_weights
                else:
                    self.request_count += 1
                    translated_cats = []
                    for cat in categories:
                        if cat == "not an item":
                            pass
                        else:
                            for i in range(len(results)):
                                try:
                                    cat_in_path = results[i]["categoryPath"].lower().find(cat["abbr"])
                                except AttributeError or KeyError:
                                    pass
                                else:
                                    if cat_in_path > -1:
                                        translated_cats.append(cat["name"])

                    if len(translated_cats) == 0:
                        return equal_weights
                    else:
                        return self._list_weights(translated_cats)

    def _select_ids(self, data_range, perc, **kwargs):
        selectby = kwargs.get("selection", "random")
        skip_ids = kwargs.get("exclude_ids", [])
        _ids = []
        if selectby == "random":
            _ids = []
            nids = int(perc*data_range)

            while nids > 0:
                rowid = random.choice(range(0, data_range))
                if rowid in skip_ids:
                    pass
                else:
                    _ids.append(rowid)
                    nids -= 1
        elif selectby == "inline":
            _ids.extend([r for r in range(0, int(data_range*perc))])
        elif selectby == "all":
            print("invalid selection method")
            print("returning all data as training")
            _ids.extend([i for i in range(0, data_range)])
        else:
            raise ValueError("Invalid selectby method, must be random, inline or all")
        return _ids

    def _get_last_id(self, file):
        mx = 0
        with open(file=file, mode="r") as f:
            readr = csv.DictReader(f, fieldnames=["id", "product_text", "retailer", "category", "clean_txt", "cat_wts"])
            mx += int(len([row for row in readr])-1)
        f.close()
        return mx
