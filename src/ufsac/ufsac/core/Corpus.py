from ufsac.common.XMLHelper import XMLHelper
from ufsac.ufsac.core.ParentLexicalEntity import ParentLexicalEntity
import xml.etree.ElementTree as ET
from ufsac.ufsac.core.Document import Document
from ufsac.ufsac.core.Word import Word
from ufsac.ufsac.core.Sentence import Sentence
from ufsac.ufsac.core.Paragraph import Paragraph


class Corpus(ParentLexicalEntity):
    def __init__(self):
        super().__init__()

    def add_document(self, document):
        self.addChild(document)

    def add_documents(self, documents):
        self.addChildren(documents)

    def get_documents(self):
        return self.getChildren()

    def get_sentences(self):
        sentences = []
        for d in self.get_documents():
            sentences.extend(d.get_sentences())
        return sentences

    def get_words(self):
        words = []
        for d in self.get_documents():
            words.extend(d.get_words())
        return words

    @staticmethod
    def load_from_xml(path):
        corpus = Corpus()
        tree = ET.parse(path)
        for d in tree.findall('document'):
            doc = Document()
            doc.set_annotation(next(iter(d.attrib.keys())), d.attrib['id'])
            corpus.add_document(doc)
            for p in d.findall('paragraph'):
                par = Paragraph()
                doc.add_paragraph(par)
                for s in p.findall('sentence'):
                    sent = Sentence()
                    sent.set_annotation(next(iter(s.attrib.keys())), s.attrib['id'])
                    par.add_sentence(sent)
                    for w in s.findall('word'):
                        attributes = w.attrib
                        w = ''
                        for k, v in attributes.items():
                            if k == 'surface_form':
                                w = Word(XMLHelper.from_valid_xml_entity(v))
                            else:
                                w.set_annotation(XMLHelper.from_valid_xml_entity(k), XMLHelper.from_valid_xml_entity(v))
                        sent.add_word(w)
        return corpus

    @staticmethod
    def loadFromXMLs(paths):
        whole = Corpus()
        for path in paths:
            part = whole.load_from_xml(path)
            whole.add_documents(part.get_documents())
        return whole
