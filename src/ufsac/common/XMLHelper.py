class XMLHelper:
    @staticmethod
    def get_indent(indent_level):
        indent = ""
        indent_size = 4
        for i in range(indent_size * indent_level):
            indent += " "
        return indent

    @staticmethod
    def to_valid_xml_entity(value):
        value_cleaned = value
        value_cleaned = value_cleaned.replace("&", "&amp;")
        value_cleaned = value_cleaned.replace("<", "&lt;")
        value_cleaned = value_cleaned.replace(">", "&gt;")
        value_cleaned = value_cleaned.replace("'", "&apos;")
        value_cleaned = value_cleaned.replace("\"", "&quot;")
        return value_cleaned

    @staticmethod
    def from_valid_xml_entity(value):
        value_cleaned = value
        value_cleaned = value_cleaned.replace("&amp;", "&")
        value_cleaned = value_cleaned.replace("&lt;", "<")
        value_cleaned = value_cleaned.replace("&gt;", ">")
        value_cleaned = value_cleaned.replace("&apos;", "'")
        value_cleaned = value_cleaned.replace("&quot;", "\"")
        return value_cleaned
