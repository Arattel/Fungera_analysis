import pickle
import os


class GenomeCache:
    def __init__(self, path: str = None):
        self.path = path
        if os.path.exists(self.path):
            self.cache = self.load_cache()
            # print(self.cache)
            # print(len(self.cache))
        else:
            self.cache = []
            self.dump_cache()

    def load_cache(self):
        with open(self.path, 'rb') as file:
            return pickle.load(file)

    def dump_cache(self):
        with open(self.path, 'wb') as file:
            pickle.dump(self.cache, file)

    def key_in_cache(self, key_to_check):
        for key in self.cache:
            if key['key'].shape == key_to_check.shape and (key['key'] == key_to_check).all():
                return True
        else:
            return False

    def set_key(self, key, value):
        self.cache.append({
            'key': key,
            'value': value
        })
        self.dump_cache()
        # print(len(self.cache))

    def get_key(self, key):
        for record in self.cache:
            if record['key'].shape == key.shape and (record['key'] == key).all():
                return record['value']
