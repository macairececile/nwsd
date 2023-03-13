from ufsac.ufsac.core.LexicalEntity import LexicalEntity


class Word(LexicalEntity):
    def __init__(self, value=None):
        if value is None:
            super().__init__()
        elif type(value) == str:
            super().__init__()
            self.set_annotation("surface_form", value)
        elif type(value) == Word():
            super().__init__(value)

    def set_value(self, value):
        self.set_annotation("surface_form", value)

    def get_value(self):
        return self.get_annotation_value("surface_form")

    def to_string(self):
        return self.get_annotation_value("surface_form")
