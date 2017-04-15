#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import json
import re
import cv2
from hockey_numbers.markup.constants import MASK_DIR, TEMPLATE_IMAGE, TEMPLATE_MASK
from .blob import filterBlobsBySize, filterBlobsByField, getBlobsFromMasks, getNearestBlob

class Marking:
    def __init__(self):
        self.dictMarkIms = {"annotation": {}}
    def addFromJson(self, jsonFile):
        newDictMarkImgs = json.load(jsonFile)
        imgObjDict = self.dictMarkIms["annotation"]
        for key, value in newDictMarkImgs["annotation"].items():
            imgObjDict[key] = value
    # получение словаря объектов по метке, например "hardly_visible"
    def getByMark(self, mark):
        imgObjDict = self.dictMarkIms["annotation"]
        subdict = {}
        for nameImage, objects in imgObjDict.items():
            objects = objects["objects"]
            markObjects = []
            for obj in objects:
                if obj.get(mark, False):
                    markObjects.append(obj)
            if len(markObjects) != 0:
                subdict[nameImage] = markObjects
        return subdict
    def filterByField(self, minHeight, maxHeight, minWidth, maxWidth):
        imgObjDict = self.dictMarkIms["annotation"]
        for nameImage, objects in imgObjDict.items():
            objs = objects["objects"]
            newobjs = []
            for obj in objs:
                rect = obj['body_rect']
                x, y, h, w = rect['x'], rect['y'], rect['h'], rect['w']
                if y >= minHeight and y + h <= maxHeight \
                        and x >= minWidth and x + w <= maxWidth:
                    newobjs.append(obj)
            objects["objects"] = newobjs
    
    def saveJson(self, outfile):
        json.dump(self.dictMarkIms, outfile, sort_keys = True)



class MarkingApproximator:
    def __init__(self):
        self.dictMarkIms = {"annotation":{}}
    def loadFromJson(self, jsonFile):
        self.dictMarkIms = json.load(jsonFile)
    def addFromJson(self, jsonFile):
        newDictMarkImgs = json.load(jsonFile)
        imgObjDict = self.dictMarkIms["annotation"]
        for key, value in newDictMarkImgs["annotation"].items():
            imgObjDict[key] = value
    def leadToStandartForm(self):
        """
         Разрешение конфликтов, типа есть номер + "номер не виден" => "номер не виден"
         "hardly_visible" заменяется на "have_no_idea", что соответвуют случаям, где bounding box
         вообще не соотвествуют хоккеистам, или номер виден, но частично
        """
        imgObjDict = self.dictMarkIms["annotation"]
        for objects in imgObjDict.values():
            objects = objects["objects"]
            for obj in objects:
                for proper in ["hardly_visible", "have_no_idea", "number_isnt_visible"]:
                    if proper in obj.keys() and obj[proper] == False:
                        obj.pop(proper)

                if "hardly_visible" in obj.keys():
                    obj.pop("hardly_visible")
                    obj["have_no_idea"] = True
                keys = obj.keys()
                if ("number" in keys) and ("have_no_idea" in keys or "number_isnt_visible" in keys):
                    obj.pop("number")
    def approximBetweenImages(self, masksDir, templateMasks, templateImages):
        self.leadToStandartForm()
        pattern = re.compile(r'\d+')
        numbFrames = map(lambda name: int(pattern.search(name).group(0)), self.dictMarkIms["annotation"].keys())
        numbFrames = sorted(numbFrames)
        framesPair = [(numbFrames[i], numbFrames[i + 1]) for i in range(len(numbFrames) - 1)]
        for first, last in framesPair:
            #получаем все промежуточные блобы
            blobsByNumber = {}
            for numb in range(first, last + 1):
                nameMask = masksDir + templateMasks.format(numb)
                mask = cv2.imread(nameMask)[:, :, 0]
                blobsByNumber[numb] = filterBlobsByField(filterBlobsBySize(getBlobsFromMasks(mask)))
            #находим последовательности блобов
            for blob1 in blobsByNumber[first]:
                chain = {first:blob1}
                curBlob = blob1
                res = True
                for nextFrame in range(first + 1, last + 1):
                    numbBlob = getNearestBlob(curBlob, blobsByNumber[nextFrame])
                    if numbBlob == -1:
                        res = False
                        break
                    curBlob = blobsByNumber[nextFrame].pop(numbBlob)
                    chain[nextFrame] = curBlob
                if res == True:
                    key1, value1 = self._getMark(templateImages.format(first), chain[first])
                    key2, value2 = self._getMark(templateImages.format(last), chain[last])
                    if key1 == key2 and value1 == value2:
                        for numbFrame in range(first + 1, last):
                            self._addNewMark(templateImages.format(numbFrame), chain[numbFrame], key1, value1)

    def saveJson(self, outfile):
        json.dump(self.dictMarkIms, outfile, sort_keys = True, indent = 4)

    def _getMark(self, nameImage, blob):
        objects = self.dictMarkIms["annotation"][nameImage]["objects"]
        keys = ["number", "have_no_idea", "number_isnt_visible"]
        for obj in objects:
            rect = obj["body_rect"]
            if blob.x == rect["x"] and blob.y == rect["y"] and blob.width == rect["w"] and blob.height == rect["h"]:
                for key in keys:
                    if key in obj.keys():
                        return key, obj[key]
        return "have_no_idea", True

    def _addNewMark(self, nameImage, blob, key, value):
        objects = self.dictMarkIms["annotation"].get(nameImage, {"objects":[]})
        objects = objects["objects"]
        objects.append({"body_rect":{"x":float(blob.x), "y":float(blob.y),
                                     "w":float(blob.width), "h":float(blob.height)},
                        key:value})
        self.dictMarkIms["annotation"][nameImage] = {"objects":objects}


class BaseMarkingCreator:
    def __init__(self, masksDir=MASK_DIR, \
                       templateMasks = TEMPLATE_MASK, \
                       templateImages = TEMPLATE_IMAGE):
        self.masksDir = masksDir
        self.templateImages = templateImages
        self.templateMasks = templateMasks
        self.blobs = {}

    def addFrames(self, frameNumbers):
        for numb in frameNumbers:
            nameMask = osp.join(self.masksDir, self.templateMasks.format(numb))
            nameImage = self.templateImages.format(numb)
            mask = cv2.imread(nameMask)
            if mask is not None:
                mask = mask[:, :, 0]
                self.blobs[nameImage] = getBlobsFromMasks(mask)
            else:
                sys.stderr.write('Mask № {:d} by path {:s} not found!\n'.format(numb, nameMask))

    def clear(self):
        self.blobs = {}

    def filterBySize(self, minHeight, maxHeight, minWidth, manWidth):
        for image in self.blobs.keys():
            self.blobs[image] = filterBlobsBySize(self.blobs[image], minHeight, maxHeight, minWidth, manWidth)

    def filterByField(self, minHeight, maxHeight, minWidth, manWidth):
        for image in self.blobs.keys():
            self.blobs[image] = filterBlobsByField(self.blobs[image], minHeight, maxHeight, minWidth, manWidth)
    
    def getBlobs(self):
        return self.blobs.copy()

    def saveAsJson(self, outFile):
        dictBlobs = {}
        for image, blobs in self.blobs.items():
            objects = []
            for blob in blobs:
                objects.append({"body_rect": {'x': float(blob.x), 'y': float(blob.y), \
                                              'h': float(blob.height), 'w': float(blob.width)}})
            if len(objects) != 0:
                dictBlobs[image] = {"objects": objects}
        json.dump(dictBlobs, outFile, sort_keys = True)

class MarkingStatictics:
    def __init__(self):
        self.dictMarking = {}

    def addJsonMarking(self, path):
        marking = Marking()
        marking.addFromJson(open(path))
        #marking.leadToStandartForm()
        dictObjsImg = marking.dictMarkIms["annotation"]
        numbs = []
        have_no_idea = 0
        number_isnt_visible = 0
        for objs in dictObjsImg.values():
            objs = objs["objects"]
            for obj in objs:
                keys = obj.keys()
                if "number_isnt_visible" in keys and obj["number_isnt_visible"]:
                    number_isnt_visible += 1
                if "have_no_idea" in keys and obj["have_no_idea"]:
                    have_no_idea += 1
                if "number" in keys and obj["number"] is not None:
                    numbs.append(int(obj["number"]))
        self.dictMarking[path] = {"number_isnt_visible":number_isnt_visible,
                                  "have_no_idea":have_no_idea,
                                  "numbers":numbs}
    def print(self, outFile):
        templateOut = "numbers: {:5d},  number_isnt_visible: {:5d}, have_no_idea: {:5d}\n"
        all_numbs = []
        all_have_no_idea = 0
        all_number_isnt_visible = 0
        for path, value in self.dictMarking.items():
            outFile.write(path + '\n')
            outFile.write(templateOut.format(len(value["numbers"]),
                                     value["number_isnt_visible"],
                                     value["have_no_idea"]))
            all_numbs.extend(value["numbers"])
            all_number_isnt_visible += value["number_isnt_visible"]
            all_have_no_idea += value["have_no_idea"]
        outFile.write("ALL\n")
        outFile.write(templateOut.format(len(all_numbs), all_number_isnt_visible, all_have_no_idea))

    def getMarkingCounts(self):
        return self.dictMarking.copy()

