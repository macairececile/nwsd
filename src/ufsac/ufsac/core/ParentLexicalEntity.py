from copy import copy

from ufsac.ufsac.core.LexicalEntity import LexicalEntity

class ParentLexicalEntity(LexicalEntity):
    def __init__(self):
        super().__init__()
        self.children = []

    def getChildren(self):
        return copy(self.children)

    def addChild(self, child):
        if child in self.children:
            return
        self.children.append(child)
        # child.setParent(this);

    def addChildren(self, children):
        childrenCopy = [children]
        for child in childrenCopy:
            self.addChild(child)

    def removeChild(self, child):
        if child not in self.children: return
        self.children.pop(child)
        # child.setParent(null);

    def removeAllChildren(self):
        childrenBefore = [self.children]
        for child in childrenBefore:
            self.removeChild(child)