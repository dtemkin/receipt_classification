import requests
from utils import config, find_location
from time import sleep
from warnings import warn

class API:
    def __init__(self, **kwargs):
        self.cfg = config()

        self.globalheaders = {"user-agent": kwargs.get("user-agent", self.cfg.WALMART["global_header"]["useragent"])}
        self.max_requests = 0
        self.request_number = 0
        self.globalpayload = {"apiKey": kwargs.get("apikey", self.cfg.WALMART["APIKEYS"][kwargs.get("api_key_nunmber", 1)]), "format": "json"}
        self.baseurl = self.cfg.WALMART["baseurl"]

    def _process_request(self, requrl, resultskey):
        req = requests.get(url=requrl, params=self.globalpayload, headers=self.globalheaders)
        if req.status_code != 200:
            js = None
        else:

            if resultskey is None:
                try:
                    js = req.json()
                except KeyError as err:
                    raise KeyError(err)
            else:
                try:
                    js = req.json()[resultskey]
                except KeyError as err:
                    raise KeyError(err)

        return js

    def _valid_kwargs(self, endpoint_kwargs, valid_keys):
        return dict(map(lambda x: (x, endpoint_kwargs.get(x)), valid_keys))

    def _current_location(self, save=True):
        try:
            loc = self.cfg.USER["location"]
        except KeyError:
            city = find_location()
            if save is True:
                self.cfg.update(section="USER", key="location", value=city)
                print("Saved")
            else:
                pass
        else:
            print("A stored location was found in configuration file.\n")
            usrin = input("Press 'y' to use config file value or 'n' to retrieve a new one. ")
            if usrin.lower() != "n":
                city = loc
            else:
                city = self.cfg.USER["location"]
        if city is None:
            raise LookupError("Could not find location using iplocation.com service\n"
                              "try rerunning request. If issue persists enter city manually")
        else:
            return city

    def Lookup(self, item_ids, **kwargs):
        '''
        :param item_ids: 
            str, list 
            required=yes
            default=none
        :param format:
            str
            required=no
            default="json"
            valid="xml","json"
        :return: 
        '''

        endpoint_url = "/".join([self.baseurl, "items?"])
        id_lists = []
        if type(item_ids) is list or tuple:
            if len(item_ids) > 20:
                warn("max number of items == 20 per request. proceeding with request but splitting the item_ids across multiple requests.")
                nitems = len(item_ids)
                for i in range(nitems, 20):
                    ids = ",".join(item_ids[0 + i, i])
                    id_lists.append(ids)
            else:
                id_lists.append(",".join([item_ids]))
        elif type(item_ids) is int:
            id_lists.append(item_ids)
        else:
            raise ValueError("item_ids must be int or list of ints")

        valid_opts = self._valid_kwargs(kwargs, valid_keys=["format"])
        self.globalpayload.update(valid_opts)
        data_store = [("categories", "categoryID_walmart",
                       "product_name", "product_upc12",
                       "product_mrsp", "seller_name")]
        for s in id_lists:
            self.globalpayload.update({"ids": s})
            req = requests.get(url=endpoint_url, params=self.globalpayload, headers=self.globalheaders)

            x = self._process_request(requrl=endpoint_url, resultskey="items")
            if x is None:
                sleep(30)
                x = self._process_request(requrl=endpoint_url, resultskey="items")
                if x is None:
                    print("HTTP Request Error")
                else:

                    data_store.append((x[0].get("categoryPath").split("/"), x[0].get("categoryNode"),
                                       x[0].get("name").replace(",", ""), str(x[0].get("upc")),
                                       float(x[0].get("msrp")), x[0].get("sellerInfo", "")))


            else:
                data_store.append((x[0].get("categoryPath").split("/"),
                                   x[0].get("name").replace(",", ""), str(x[0].get("upc")),
                                   float(x[0].get("msrp")), x[0].get("sellerInfo", "")))
        return dict(data_store)

    def CategoryPathExtraction(self, category_id):
        category_parts = category_id.split("_")
        categories = self.Taxonomy()
        parent = dict(filter(lambda x: x["id"] == category_parts[0], [i for i in categories]))
        parent_name = parent["name"]
        major_category = dict(filter(lambda x: x["id"] == "_".join([category_parts[0], category_parts[1]]),
                                     [i for i in parent["children"]]))
        major_category_name = major_category["name"]
        category = dict(filter(lambda x: x["id"] == id,
                               [i for i in major_category["children"]]))
        category_name = category["name"]
        return [parent_name, major_category_name, category_name]

    def Search(self, qstring, **kwargs):
        '''

        :param qstring: 
        :param kwargs: 
        :return: 
        '''
        endpoint_url = "/".join([self.baseurl, "search?"])
        valid_opts = self._valid_kwargs(endpoint_kwargs=kwargs,
                                        valid_keys=["format", "categoryId",
                                                    "facet", "facetfilter",
                                                    "facetrange"])
        self.globalpayload.update(dict(query=qstring, **valid_opts))
        d = self._process_request(requrl=endpoint_url, resultskey="items")
        fields = kwargs.get("fields", ["name", "upc", "salePrice",
                                       "categoryPath", "categoryNode",
                                       "customerRating"])
        dd = []
        for dct in d:
            dd.append(self._select_fields(dct=dct, fields=fields))
        return dd

    def _select_fields(self, dct, fields):
        d = dict(map(lambda x: (x, 0), fields))
        shared = set([k for k in dct.keys()]).intersection(set(fields))
        for x in shared:
            d.update({x: dct[x]})
        return d

    def Taxonomy(self, **kwargs):
        endpoint_url = "/".join([self.baseurl, "taxonomy?"])
        valid_opts = self._valid_kwargs(kwargs, ["format"])
        self.globalpayload.update(valid_opts)
        d = self._process_request(requrl=endpoint_url, resultskey="categories")
        return d

    def Reviews(self, item_id, **kwargs):
        endpoint_url = "/".join([self.baseurl, "reviews", item_id + "?"])
        valid_opts = self._valid_kwargs(kwargs, ["format"])
        self.globalpayload.update(valid_opts)
        d = self._process_request(requrl=endpoint_url, resultskey="reviews")
        return d

    def Trending(self, **kwargs):
        endpoint_url = "/".join([self.baseurl, "trends"])
        valid_opts = self._valid_kwargs(kwargs, ["format"])
        self.globalpayload.update(valid_opts)
        d = self._process_request(requrl=endpoint_url, resultskey="items")
        return d

    def Paginated(self, **kwargs):
        '''
        :kwargs format str optional:
            default="json" 
            accepts=["json","xml"]
        :kwargs category str optional:
        :kwargs brand str optional:
        :kwargs specialOffer str optional:
        :return JSONObject: 
        '''
        endpoint_url = "/".join([self.baseurl, "paginated", "items?"])
        valid_kwargs = self._valid_kwargs(kwargs, ["format", "category", "brand", "specialOffer"])
        self.globalpayload.update(valid_kwargs)
        d = self._process_request(requrl=endpoint_url, resultskey="items")
        return d

    def StoreLocator(self, city="auto-detect", **kwargs):
        '''
        :format str optional:
            default="json"
            valid="json, xml"
        :lat str optional:
        :long str optional:
        :zip str optional:
        :city str optional:
        :return JSONObject: 
        '''

        if city == "auto-detect":
            loc = self._current_location()
        else:
            loc = city
        endpoint_url = "/".join([self.baseurl, "stores?"])
        valid_kwargs = dict(map(lambda x: (x, kwargs.get(x)), ["format", "lat", "long", "zip"]))
        self.globalpayload.update(valid_kwargs)
        self.globalpayload.update({"city": loc})
        req = requests.get(url=endpoint_url, params=self.globalpayload, headers=self.globalheaders)
        return req.json()



    def DailyValue(self, **kwargs):
        '''
        ::
        :return: 
        '''
        pass
