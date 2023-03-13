import statistics


class MultipleDisambiguationResult:
    def __init__(self):
        self.results = []

    def add_disambiguation_result(self, result):
        self.results.append(result)

    def score_mean(self):
        return statistics.mean(self.all_scores())

    def score_standard_deviation(self):
        if len(self.all_scores()) < 2:
            return 0
        else:
            return statistics.stdev(self.all_scores(), self.score_mean())

    def time_mean(self):
        return statistics.mean(self.all_times())

    def all_scores(self):
        return [r.score_f1() for r in self.results]

    def all_times(self):
        return [r.time for r in self.results]
