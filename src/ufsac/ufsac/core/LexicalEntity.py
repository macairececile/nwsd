from ufsac.ufsac.core.Annotation import *


class LexicalEntity:
    def __init__(self, lexical_entity_to_copy=None):
        self.annotations_as_list = []
        self.annotations_as_map = {}
        if lexical_entity_to_copy:
            for annotation_to_copy in lexical_entity_to_copy.get_annotations():
                self.set_annotation(annotation_to_copy.get_annotation_name(), annotation_to_copy.get_annotation_value())

    def get_annotations(self):
        return list(self.annotations_as_list)

    def get_annotation_value(self, annotation_name: str):
        if annotation_name not in self.annotations_as_map.keys():
            return ""
        return self.annotations_as_map[annotation_name].get_annotation_value()

    def get_annotation_values(self, annotation_name: str, delimiter: str):
        if annotation_name not in self.annotations_as_map.keys():
            return []
        return self.annotations_as_map[annotation_name].get_annotation_values(delimiter)

    def set_annotation(self, annotation_name: str, annotation_value, delimiter=None):
        is_str = False
        is_list = False
        if annotation_name is None or annotation_name == "":
            return
        if annotation_value is None:
            if isinstance(annotation_value, str):
                is_str = True
                annotation_value = ""
            if isinstance(annotation_value, list):
                is_list = True
                annotation_value = []
        else:
            if isinstance(annotation_value, str):
                is_str = True
            if isinstance(annotation_value, list):
                is_list = True
        if self.has_annotation(annotation_name):
            if is_str:
                self.annotations_as_map[annotation_name].set_annotation_value(annotation_value)
            if is_list:
                self.annotations_as_map[annotation_name].set_annotation_values(annotation_value, delimiter)
        else:
            if is_str:
                a = Annotation(annotation_name, annotation_value)
                self.annotations_as_list.append(a)
                self.annotations_as_map[annotation_name] = a
            if is_list:
                a = Annotation(annotation_name, annotation_value, delimiter)
                self.annotations_as_list.append(a)
                self.annotations_as_map[annotation_name] = a

    def remove_annotation(self, annotation_name: str):
        self.annotations_as_list = [x for x in self.annotations_as_list if annotation_name != x.get_annotation_name()]
        self.annotations_as_map.pop(annotation_name, None)

    def remove_all_annotations(self):
        self.annotations_as_list = []
        self.annotations_as_map = {}

    def has_annotation(self, annotation_name: str):
        return bool(self.get_annotation_value(annotation_name))

    def transfert_annotations_to_copy(self, copy):
        for a in self.annotations_as_list:
            copy.set_annotation(a.get_annotation_name(), a.get_annotation_value())
