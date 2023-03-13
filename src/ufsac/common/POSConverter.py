class POSConverter:
    @staticmethod
    def to_wn_pos(any_pos):
        if isinstance(any_pos, str):
            any_pos = any_pos.lower()
            if any_pos.startswith("n"):
                return "n"
            if any_pos.startswith("v"):
                return "v"
            if any_pos.startswith("r") or any_pos.startswith("adv"):
                return "r"
            if any_pos.startswith("j") or any_pos.startswith("a"):
                return "a"
            return "x"
        elif isinstance(any_pos, int):
            if any_pos == 1:
                return "n"
            if any_pos == 2:
                return "v"
            if any_pos == 3:
                return "a"
            if any_pos == 4:
                return "r"
            if any_pos == 5:
                return "a"
            return "x"

    @staticmethod
    def to_ptb_pos(any_pos):
        any_pos = POSConverter.to_wn_pos(any_pos)
        if any_pos == "n":
            return "NN"
        if any_pos == "v":
            return "VB"
        if any_pos == "a":
            return "JJ"
        if any_pos == "r":
            return "RB"
        return ""

    all_ptb_pos = {"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
                 "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
                 "VBZ", "WDT", "WP", "WP$", "WRB"}

    @staticmethod
    def is_ptb_pos(any_pos):
        return any_pos in POSConverter.all_ptb_pos
