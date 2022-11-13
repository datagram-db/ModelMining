import pandas as pd

from dataloading.Log import Log
from embeddings.declare.DatalessDeclare import DatalessDeclare
from embeddings.declare.Utils import DeclareCandidate


class DeclareDevMining:
    def __init__(self):
        self.ddcl = DatalessDeclare()

    def filter_candidates_by_support(self, candidates, logPos, logNeg, support_norm, support_dev):
        assert isinstance(logPos, Log)
        assert isinstance(logNeg, Log)
        filtered_candidates = []
        for candidate in candidates:
            assert isinstance(candidate, DeclareCandidate)
            count_dev = 0
            count_norm = 0
            count_dev, count_norm, filtered_candidates = self.iterateOverLogForSupport(candidate, count_dev, count_norm,
                                                                                       filtered_candidates, 1,
                                                                                       logPos, support_dev,
                                                                                       support_norm)
            count_dev, count_norm, filtered_candidates = self.iterateOverLogForSupport(candidate, count_dev, count_norm,
                                                                                       filtered_candidates, 0,
                                                                                       logNeg, support_dev,
                                                                                       support_norm)
        return filtered_candidates

    def iterateOverLogForSupport(self, candidate, count_dev, count_norm, filtered_candidates, label, logPos,
                                 support_dev,
                                 support_norm):
        for i in range(logPos.getNTraces()):
            trace = logPos.getIthTrace(i)
            ev_ct = 0
            for event in candidate.args:
                if trace.hasEvent(event):
                    ev_ct += 1
                else:
                    break
            if ev_ct == len(candidate):  # all candidate events in trace
                if label == 1:
                    count_dev += 1
                else:
                    count_norm += 1

            if count_dev >= support_dev or count_norm >= support_norm:
                filtered_candidates.append(candidate)
                break
        return count_dev, count_norm, filtered_candidates

    def run(self, logPos, logNeg, candidates=None, filterCandidates=True,
             candidate_threshold=0.1, constraint_threshold=0.1):
        assert isinstance(logPos, Log)
        assert isinstance(logNeg, Log)
        doTest = candidates is None
        finalCandidates = []
        if doTest:
            print("Log: setting the candidates")
            events_set = set.union(logPos.unique_events, logNeg.unique_events)
            candidates = [DeclareCandidate([event]) for event in events_set] + [DeclareCandidate([e1, e2]) for e1 in
                                                                                events_set for e2 in events_set if e1 != e2]
            normal_count = logPos.getNTraces()
            deviant_count = logNeg.getNTraces()
            ev_support_norm = int(normal_count * candidate_threshold)
            ev_support_dev = int(deviant_count * candidate_threshold)
            constraint_support_dev = int(deviant_count * constraint_threshold)
            constraint_support_norm = int(normal_count * constraint_threshold)
            if filterCandidates:
                print("Filtering the candidate for the templates")
                candidates = self.filter_candidates_by_support(candidates, logPos, logNeg, ev_support_norm, ev_support_dev)
            candidates = self.ddcl.instantiateDeclares(candidates)
        all_results = {}
        for x in candidates:
            constraint_result = x.toArray(logPos, logNeg)
            fulfill_norm = 0
            fulfill_dev = 0
            N = 0
            for i in range(logPos.getNTraces()):
                if constraint_result[i] > 0:
                    fulfill_dev += 1
            N = logPos.getNTraces()
            for i in range(logNeg.getNTraces()):
                if constraint_result[N+i] > 0:
                    fulfill_norm += 1
            norm_pass = True
            dev_pass = True
            if doTest:
                norm_pass = fulfill_norm >= constraint_support_norm
                dev_pass = fulfill_dev >= constraint_support_dev
            satis_normal, satis_deviant  = norm_pass, dev_pass
            if not doTest or (not filterCandidates or (satis_normal or satis_deviant)):
                all_results[x.__str__()] = constraint_result
                if doTest:
                    finalCandidates.append(x)
        if not doTest:
            finalCandidates = candidates
        all_results["Class"] = ([1] *logPos.getNTraces())+([0]*logNeg.getNTraces())
        merged = pd.DataFrame(all_results)
        merged = merged.fillna(0)
        return merged, finalCandidates




