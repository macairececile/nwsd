class Annotation:
    def __init__(self, annotation_name, annotation_value, delimiter=None):
        if annotation_name is None:
            self.annotation_name = ""
        else:
            self.annotation_name = annotation_name
        if annotation_value is None:
            self.annotation_value = ""
        else:
            if type(annotation_value) == str:
                self.annotation_value = annotation_value
            elif type(annotation_value) == list:
                self.annotation_value = delimiter.join(annotation_value)

    def get_annotation_name(self):
        return self.annotation_name

    def get_annotation_value(self):
        return self.annotation_value

    def set_annotation_value(self, value: str):
        if value == "":
            self.annotation_value = ""
        else:
            self.annotation_value = value

    def get_annotation_values(self, delimiter: str):
        return self.annotation_value.split(delimiter)

    def set_annotation_values(self, values: list, delimiter: str):
        if not values:
            self.annotation_value = ""
        else:
            self.annotation_value = delimiter.join(values)

    def __str__(self):
        return self.annotation_name + "=" + self.annotation_value
