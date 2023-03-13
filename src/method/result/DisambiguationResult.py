class DisambiguationResult:
    def __init__(self, total=0, good=0, bad=0, total_per_pos=None, good_per_pos=None, bad_per_pos=None):
        self.total = total
        self.good = good
        self.bad = bad
        if total_per_pos is None:
            self.total_per_pos = DisambiguationResult.init_map_per_pos()
        else:
            self.total_per_pos = total_per_pos
        if good_per_pos is None:
            self.good_per_pos = DisambiguationResult.init_map_per_pos()
        else:
            self.good_per_pos = good_per_pos
        if bad_per_pos is None:
            self.bad_per_pos = DisambiguationResult.init_map_per_pos()
        else:
            self.bad_per_pos = bad_per_pos

    def concatenate_result(self, other):
        self.total += other.total
        self.good += other.good
        self.bad += other.bad
        for pos in ["n", "v", "a", "r", "x"]:
            self.total_per_pos[pos] = self.total_per_pos[pos] + other.total_per_pos[pos]
            self.good_per_pos[pos] = self.good_per_pos[pos] + other.good_per_pos[pos]
            self.bad_per_pos[pos] = self.bad_per_pos[pos] + other.bad_per_pos[pos]

    def attempted(self):
        return self.good + self.bad

    def missed(self):
        return self.total - self.attempted()

    def coverage(self):
        return self.ratio_percent(self.attempted(), self.total)

    def score_recall(self):
        return self.ratio_percent(self.good, self.total)

    def score_precision(self):
        return self.ratio_percent(self.good, self.attempted())

    def score_f1(self):
        r = self.score_recall()
        p = self.score_precision()
        if p + r == 0:
            return 0
        else:
            return 2.0 * ((p * r) / (p + r))

    def attempted_per_pos(self, pos):
        return self.good_per_pos[pos] + self.bad_per_pos[pos]

    def missed_per_pos(self, pos):
        return self.total_per_pos[pos] - self.attempted_per_pos(pos)

    def coverage_per_pos(self, pos):
        return self.ratio_percent(self.attempted_per_pos(pos), self.total_per_pos[pos])

    def score_recall_per_pos(self, pos):
        return self.ratio_percent(self.good_per_pos[pos], self.total_per_pos[pos])

    def score_precision_per_pos(self, pos):
        return self.ratio_percent(self.good_per_pos[pos], self.attempted_per_pos(pos))

    def score_f1_per_pos(self, pos):
        r = self.score_recall_per_pos(pos)
        p = self.score_precision_per_pos(pos)
        if p + r == 0:
            return 0
        else:
            return 2.0 * ((p * r) / (p + r))

    @staticmethod
    def ratio_percent(num, den):
        if den == 0:
            return 0
        else:
            return (num / den) * 100

    @staticmethod
    def init_map_per_pos():
        map_pos = {}
        for pos in ["n", "v", "a", "r", "x"]:
            map_pos[pos] = 0
        return map_pos
