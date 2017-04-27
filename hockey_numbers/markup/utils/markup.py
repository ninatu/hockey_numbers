from collections import defaultdict
from copy import deepcopy
import json

class FrameObject:
    def __init__(self, x, y, w, h, data={}):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._data = data

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def data(self):
        return deepcopy(self._data)

    @staticmethod
    def create_from_json(json_data):
        body_rect = json_data['body_rect']
        x = body_rect['x']
        y = body_rect['y']
        w = body_rect['w']
        h = body_rect['h']
        data = {}
        for key, val in json_data.items():
            if key != 'body_rect':
                data[key] = val
        return FrameObject(x, y, w, h, data)

    def to_json(self):
        obj_dict = deepcopy(self._data)
        obj_dict['body_rect'] = { 'x': int(self.x),
                                  'y': int(self.y),
                                  'w': int(self.w),
                                  'h': int(self.h)}
        return obj_dict


class Frame:
    def __init__(self):
        self._objects = []
        self._data = {}

    def add_obj(self, obj):
        self._objects.append(obj)

    def add_data(self, data):
        for key, val in data.items():
            if key != 'objects':
                self._data[key] = val

    @property
    def objects(self):
        return deepcopy(self._objects)

    def to_json(self):
        frame_dict = {}
        objects = []
        for obj in self._objects:
            objects.append(obj.to_json())
        frame_dict['objects'] = objects

        return frame_dict

    @staticmethod
    def create_from_json(json_data):
        frame = Frame()
        for obj in json_data.get('objects', []):
            frame.add_obj(FrameObject.create_from_json(obj))

        frame.add_data(json_data)

        return frame


class Markup:

    def __init__(self):
        self._frames = defaultdict(lambda : Frame())

    # TODO merge frame data
    def merge(self, path):
        with open(path) as fin:
            json_file = json.load(fin)
            img_dict = json_file['annotation']
            for frame_name, frame_data in img_dict.items():
                self._frames[frame_name] = Frame.create_from_json(frame_data)


    def add_blob(self, frame_name, x, y, w, h, data={}):
        self._frames[frame_name].add_obj(FrameObject(x, y, w, h, data))


    def save(self, out_path):
        with open(out_path, 'w') as fout:
            json.dump(self.to_json(), fout)

    def print_statistics(self, out_file):

        keys = ["number", "number_isnt_visible", "hardly_visible"]
        counts = defaultdict(lambda : 0)

        for frame in self._frames.values():
            for obj in frame.objects:
                for key in keys:
                    if obj.data.get(key, False):
                        counts[key] += 1

        with open(out_file, 'w') as fout:
            json.dump(counts, fout)



    def to_json(self):
        img_dict = {}
        for frame_name, frame in self._frames.items():
            img_dict[frame_name] = frame.to_json()

        return img_dict
