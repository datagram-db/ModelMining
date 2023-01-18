from dataloading.Log import TracePositional
from embeddings.declare.Utils import DeclareUnary, DeclareBinary, DeclareCandidate

class Absence(DeclareUnary):
    def __init__(self, n, arg1=""):
        super().__init__("Absence", n, arg1, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count > self.n:
            return -(count - self.n), [], trace.getEventsInPositionalTrace(self.arg1)
        else:
            return 1, trace.getEventsInPositionalTrace(self.arg1), []


class Exists(DeclareUnary):
    def __init__(self, n=1, arg1=""):
        super().__init__("Exists", n, arg1, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count >= self.n:
            return count, trace.getEventsInPositionalTrace(self.arg1), []
        else:
            return -1, [], trace.getEventsInPositionalTrace(self.arg1)


class Exactly(DeclareUnary):
    def __init__(self, n, arg1=""):
        super().__init__("Exactly", n, arg1, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count == self.n:
            return 1, trace.getEventsInPositionalTrace(self.arg1), []
        else:
            return -1, [], trace.getEventsInPositionalTrace(self.arg1)


class Init(DeclareUnary):
    def __init__(self, arg1=""):
        super().__init__("Init", 1, arg1, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.length == 0) or (not trace.hasPos(self.arg1, 0)):
            return -1, [], [0]
        else:
            return 1, [0], []


class End(DeclareUnary):
    def __init__(self, arg1=""):
        super().__init__("End", 1, arg1, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.length == 0):
            return -1, [], [0]
        else:
            return 1, [trace.length-1], []

class ExclChoice(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ExclChoice", arg1, arg2, True)
        self.ex1 = Exists(1, arg1)
        self.ex2 = Exists(1, arg2)

    def __call__(self, trace):
        a, f1, v1 = self.ex1(trace)
        b, f2, v2 = self.ex2(trace)
        if (a == -1):
            if (b == -1):
                return -1, [], list(set.union(set(v1), set(v2)))
            else:
                return b, f2, []
        else:
            if (b == -1):
                return a, f1, []
            else:
                return -1, [], list(set.union(set(f1), set(f2)))


class Choice(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Choice", arg1, arg2, True)
        self.ex1 = Exists(1, arg1)
        self.ex2 = Exists(1, arg2)

    def __call__(self, trace):
        a, f1, v1 = self.ex1(trace)
        b, f2, v2 = self.ex2(trace)
        if (a == -1) and (b == -1):
            return -1, [], list(set.union(set(v1), set(v2)))
        else:
            if a == -1:
                a = 0
            if b == -1:
                b = 0
            return a + b, list(set.union(set(f1), set(f2))), []


class AltPrecedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltPrecedence", arg1, arg2, True)

    def __call__(self, trace : TracePositional):
        fulfillments = []
        violations = []
        if trace.hasEvent(self.arg2):
            if trace.hasEvent(self.arg1):
                # Go through two lists, one by one
                # first events pos must be before 2nd lists first pos etc...
                # A -> A -> B -> A -> B

                # efficiency check
                event_1_count = trace.eventCount(self.arg1)
                event_2_count = trace.eventCount(self.arg2)

                # There has to be more or same amount of event A's compared to B's
                if event_2_count > event_1_count:
                    return 0, [], []

                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

                # FOR every EVENT 2: Going before in the log: THERE MUST BE EVENT 1 BEFORE EVENT 2!
                # Keep track of largest event_1_pos before current event_2_pos. Sorting array.
                merged = []
                event_1_ind = 0
                event_2_ind = 0
                while event_1_ind < len(event_1_positions) and event_2_ind < len(event_2_positions):
                    if event_1_positions[event_1_ind] < event_2_positions[event_2_ind]:
                        merged.append((1, event_1_positions[event_1_ind]))
                        event_1_ind += 1
                    else:
                        merged.append((2, event_2_positions[event_2_ind]))
                        event_2_ind += 1

                # Merge leftovers
                while event_1_ind < len(event_1_positions):
                    merged.append((1, event_1_positions[event_1_ind]))
                    event_1_ind += 1

                while event_2_ind < len(event_2_positions):
                    merged.append((2, event_2_positions[event_2_ind]))
                    event_2_ind += 1

                # Go through array, at every point check if (2, x). If 2, then check if previous is 2 or 1.
                for i in range(len(merged)):
                    if merged[i][0] == 2:
                        if i == 0:
                            # If first one, then previous cant be 1. violation.
                            violations.append(merged[i][1])
                        elif merged[i - 1][0] == 1:
                            # If is not same, then no violation
                            fulfillments.append(merged[i][1])
                        else:
                            # Therefore if previous is same... then violation
                            violations.append(merged[i][1])

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                # impossible because there has to be at least one event1 with event2. Therefore all activations are violated
                return -1, [], trace.getEventsInPositionalTrace(self.arg2)
        return 0, [], []


class AltResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltResponse", arg1, arg2, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        fulfillments = []
        violations = []
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                # Go through two lists, one by one
                # first events pos must be before 2nd lists first pos etc...
                # A -> A -> B -> A -> B

                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)
                # FOR every EVENT 2: Going before in the log: THERE MUST BE EVENT 1 BEFORE EVENT 2!
                # Keep track of largest event_1_pos before current event_2_pos. Sorting array.
                merged = []
                event_1_ind = 0
                event_2_ind = 0
                while event_1_ind < len(event_1_positions) and event_2_ind < len(event_2_positions):
                    if event_1_positions[event_1_ind] < event_2_positions[event_2_ind]:
                        merged.append((1, event_1_positions[event_1_ind]))
                        event_1_ind += 1
                    else:
                        merged.append((2, event_2_positions[event_2_ind]))
                        event_2_ind += 1

                # Merge leftovers
                while event_1_ind < len(event_1_positions):
                    merged.append((1, event_1_positions[event_1_ind]))
                    event_1_ind += 1

                while event_2_ind < len(event_2_positions):
                    merged.append((2, event_2_positions[event_2_ind]))
                    event_2_ind += 1

                # Go through array, at every point check if (2, x). If 2, then check if next is 2 or 1.
                for i in range(len(merged)):
                    if merged[i][0] == 1:
                        if i == len(merged) - 1:
                            # Last in list! Will not be responded! Violated.
                            violations.append(merged[i][1])
                        elif merged[i + 1][0] == 2:
                            # If no violation is not same, then no violation
                            fulfillments.append(merged[i][1])
                        else:
                            # Therefore if previous is same... then violation
                            violations.append(merged[i][1])

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                # impossible because there has to be at least one event2 with event1. Therefore all activations are violated
                return -1, [],  trace.getEventsInPositionalTrace(self.arg1)
        return 0, [], []  # todo: vacuity condition!!


class AltSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltSuccession", arg1, arg2, False)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.hasEvent(self.arg1)) != (trace.hasEvent(self.arg2)):
            return -1, [], []

        if trace.hasEvent(self.arg1) and trace.hasEvent(self.arg2):
            event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

            if len(event_1_positions) != len(event_2_positions):
                return -1, [], []  # impossible if not same length

            pos = -1
            current_ind = 0
            switch = False
            while current_ind < len(event_1_positions):

                # Use switch to know from which array to get next..
                if switch:
                    next_pos = event_2_positions[current_ind]
                    current_ind += 1
                else:
                    next_pos = event_1_positions[current_ind]

                if next_pos <= pos:
                    return -1, [], []  # next one is smaller than current

                pos = next_pos  # go to next one.
                switch = not switch  # swap array

            count = len(event_1_positions)
            return count, [], []

        return 0, [], []  # vacuity condition


class ChainPrecedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainPrecedence", arg1, arg2, True)

    def __call__(self, trace : TracePositional):
        assert isinstance(trace, TracePositional)
        fulfillments = []
        violations = []
        if trace.hasEvent(self.arg2):
            if trace.hasEvent(self.arg1):
                # Each event1 must instantly be followed by event2
                event_1_positions = set(trace.getEventsInPositionalTrace(self.arg1))
                event_2_positions = set(trace.getEventsInPositionalTrace(self.arg2))
                # For every pos2, check if pos2-1 is in event1 set
                for pos2 in event_2_positions:
                    if pos2 - 1 not in event_1_positions:
                        # Then no possible to preceed!!
                        violations.append(pos2)
                    else:
                        fulfillments.append(pos2)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                return -1, [], trace.getEventsInPositionalTrace(self.arg2)  # no response for event1
        return 0, [], []  # todo, vacuity


class ChainResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainResponse", arg1, arg2, True)
    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        fulfillments = []
        violations = []
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                event_1_positions = set(trace.getEventsInPositionalTrace(self.arg1))
                event_2_positions = set(trace.getEventsInPositionalTrace(self.arg2))

                for pos1 in event_1_positions:
                    if pos1 + 1 not in event_2_positions:
                        # Exists event 1 which is not chained by event 2
                        violations.append(pos1)
                    else:
                        fulfillments.append(pos1)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                return -1, [], trace.getEventsInPositionalTrace(self.arg1)  # no response for event1
        return 0, [], []  # todo, vacuity


class ChainSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainSuccession", arg1, arg2, False)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.hasEvent(self.arg1)) != (trace.hasEvent(self.arg2)):
            return -1, [], []

        if (trace.hasEvent(self.arg1)) and (trace.hasEvent(self.arg2)):
            event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)
            if len(event_1_positions) != len(event_2_positions):
                # has to be same number of events
                return -1, [], []

            # They have to appear together, with event1 always before event2
            for i in range(len(event_1_positions)):
                if event_1_positions[i] + 1 != event_2_positions[i]:
                    return -1, [], []

            count = len(event_1_positions)
            return count, [], []

        return 0, [], []  # todo vacuity


class Succession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Succession", arg1, arg2, False)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if self.arg1 in trace and not self.arg2 in trace:
            return -1, [], []

        if self.arg2 in trace and not self.arg1 in trace:
            return -1, [], []

        if self.arg1 in trace and self.arg2 in trace:

            # First position of A
            first_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[0]

            # First position of B
            first_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[0]

            # Last position A
            last_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[-1]

            # Last position B
            last_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[-1]

            if first_pos_event_1 < first_pos_event_2 and last_pos_event_1 < last_pos_event_2:
                # todo: check frequency!
                return min(trace.eventCount(self.arg1), trace.eventCount(self.arg2)), [], []
            else:
                return -1, [], []

        # todo: vacuity condition!
        return 0, [], []


class RespExistence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("RespExistence", arg1, arg2, True)
    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                return trace.eventCount(self.arg1), trace.getEventsInPositionalTrace(self.arg1), []
            else:
                return -1, [], trace.getEventsInPositionalTrace(self.arg2)
        return 0, [], []


class Response(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Response", arg1, arg2, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        violations = []
        fulfillments = []
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                last_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[-1]

                for pos_1 in event_1_positions:
                    if pos_1 < last_pos_event_2:
                        fulfillments.append(pos_1)
                    else:
                        violations.append(pos_1)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations
            else:
                # impossible for event 2 to be after event 1 if there is no event 2
                return -1, [], trace.getEventsInPositionalTrace(self.arg1)
        return 0, [], []  # not vacuity atm..

class Precedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Precedence", arg1, arg2, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        violations = []
        fulfillments = []
        if trace.hasEvent(self.arg2):
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)
            if trace.hasEvent(self.arg1):
                first_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[0]
                # All event B's, which are before first event A are violated. Every other is fulfilled.

                for event_2_pos in event_2_positions:
                    if event_2_pos < first_pos_event_1:
                        # There is event
                        violations.append(event_2_pos)
                    else:
                        fulfillments.append(event_2_pos)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                # impossible because there has to be at least one event1 with event2
                return -1, [], event_2_positions

        # Vacuously fulfilled
        return 0, [], []

class NotSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotSuccession", arg1, arg2, False)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):

                # for this to be true, last event 2 has to be before first event 1
                first_event_1 =  trace.getEventsInPositionalTrace(self.arg1)[0]
                last_event_2 =  trace.getEventsInPositionalTrace(self.arg2)[-1]

                if first_event_1 < last_event_2:
                    return -1, [], []  # in this case there is an event 2, which appears after first event
                else:
                    return 1, [], []
            else:
                return 1, [], [] # not possible

        # if not, then impossible and template fulfilled
        return 0, [], []  # vacuity

class NotChainSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotChainSuccession", arg1, arg2, False)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1) and trace.hasEvent(self.arg2):
            # Find a place, where A and B are next
            event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

            e1_ind = 0
            e2_ind = 0
            while True:
                if e1_ind >= len(event_1_positions) or e2_ind >= len(event_2_positions):
                    return 1, [], []  # no more choices

                current_e1 = event_1_positions[e1_ind]
                current_e2 = event_2_positions[e2_ind]

                if current_e1 > current_e2:
                    e2_ind += 1
                else:
                    if current_e1 + 1 == current_e2:
                        return -1, [], []  # found a place, where they are together
                    e1_ind += 1

        # How to do vacuity here? 1 by default most likely
        return 0, [], []  # TODO, this condition?


class NotChainResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotChainResponse", arg1, arg2, True)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        violations = []
        fulfillments = []
        if trace.hasEvent(self.arg1) :
            if trace.hasEvent(self.arg2):
                # Each event1 must instantly be followed by event2
                event_1_positions = set(trace.getEventsInPositionalTrace(self.arg1))
                event_2_positions = set(trace.getEventsInPositionalTrace(self.arg2))

                # for each event_1_position, check if there is event_2_position after
                for pos1 in event_1_positions:
                    if pos1 + 1 in event_2_positions:
                        # this means that there is!
                        violations.append(pos1)
                    else:
                        fulfillments.append(pos1)

                if len(violations) > 0:
                    # at least one violation,
                    return -1, fulfillments, violations
                else:
                    # None of pos1+1 was in event2, therefore there is no chain_response
                    return len(fulfillments), fulfillments, violations

            else:
                return trace.eventCount(self.arg1), trace.getEventsInPositionalTrace(self.arg1), []  # no response for event1, therefore all fulfilled

        return 0, [], []  # todo, vacuity


class NotRespExistence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotRespExistence", arg1, arg2, True)

    def __call__(self, trace : TracePositional):
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                return trace.eventCount(self.arg1), trace.getEventsInPositionalTrace(self.arg1), []  # Third one is the place of activations!
            else:
                # Return all indices as violations!
                # violation at every activation!
                return -1, [], trace.getEventsInPositionalTrace(self.arg2)
        else:
            # Vacuous if no event_1
            return 0, [], []


class NotResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotResponse", arg1, arg2, True)

    def __call__(self, trace : TracePositional):
        violations = []
        fulfillments = []
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                last_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[-1]

                # Go through all event 1, if it is after last_pos_event_2 then fulfilled, otherwise false

                for pos1 in event_1_positions:
                    if pos1 < last_pos_event_2:
                        violations.append(pos1)
                    else:
                        fulfillments.append(pos1)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations

            else:
                # impossible for event 2 to be after event 1 if there is no event 2
                return trace.eventCount(self.arg1), trace.getEventsInPositionalTrace(self.arg1), []

        return 0, [], []  # not vacuity atm..


class NotPrecedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotPrecedence", arg1, arg2, True)

    def __call__(self, trace : TracePositional):
        fulfillments = []
        violations = []
        if trace.hasEvent(self.arg2):
            if not trace.hasEvent(self.arg1):
                return trace.eventCount(self.arg2), trace.getEventsInPositionalTrace(self.arg2), []  # Then all are fulfillments
            else:
                first_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[0]  # first A position
                # Check every pos_2
                for pos_event_2 in trace.getEventsInPositionalTrace(self.arg2):
                    if pos_event_2 < first_pos_event_1:
                        # pos_event_1 is AFTER B, therefore precedence does not hold
                        fulfillments.append(pos_event_2)
                    else:
                        # pos_event_1 is BEFORE B, therefore precedence holds!
                        violations.append(pos_event_2)

                if len(violations) > 0:
                    return -1, fulfillments, violations
                else:
                    return len(fulfillments), fulfillments, violations
        else:
            # Vacuous if no event_2
            return 0, [], []


class DatalessDeclare:
    def __init__(self):
        self.unary_template_map = [
            Init(),
            End(),
             Exists(),
             Absence(1),
             Absence(2),
             Absence(3),
             Exactly(1),
             Exactly(2),
             Exactly(3)
        ]
        self.binary_template_map = [
            ExclChoice(),
            Choice(),
             AltPrecedence(),
             AltSuccession(),
             AltResponse(),
             ChainPrecedence(),
             ChainResponse(),
            ChainSuccession(),
             NotChainSuccession(),
             NotSuccession(),
            RespExistence(),
             Response(),
             Succession(),
             Precedence(),
            NotRespExistence(),
            NotChainResponse(),
            NotResponse(),
            NotPrecedence()
        ]


    def instantiateDeclares(self, candidates):
        ls = []
        for dc in candidates:
            assert isinstance(dc, DeclareCandidate)
            if len(dc.args)==1:
                ls.extend([Init(dc.args[0]),
             Exists(1,dc.args[0]),
             Absence(1,dc.args[0]),
             Init(dc.args[0]),
             End(dc.args[0])
             # Absence(2,dc.args[0]),
             # Absence(3,dc.args[0]),
             # Exactly(1,dc.args[0]),
             # Exactly(2,dc.args[0]),
             # Exactly(3,dc.args[0])]
                #
                ])
            elif len(dc.args)==2:
                ls.extend([ExclChoice(dc.args[0], dc.args[1]),
                           Choice(dc.args[0], dc.args[1]),
                           # AltPrecedence(dc.args[0], dc.args[1]),
                           # AltSuccession(dc.args[0], dc.args[1]),
                           # AltResponse(dc.args[0], dc.args[1]),
                           ChainPrecedence(dc.args[0], dc.args[1]),
                           ChainResponse(dc.args[0], dc.args[1]),
                           # ChainSuccession(dc.args[0], dc.args[1]),
                           # NotChainSuccession(dc.args[0], dc.args[1]),
                           NotSuccession(dc.args[0], dc.args[1]),
                           RespExistence(dc.args[0], dc.args[1]),
                           Response(dc.args[0], dc.args[1]),
                           Succession(dc.args[0], dc.args[1]),
                           Precedence(dc.args[0], dc.args[1]),
                # NotRespExistence(dc.args[0], dc.args[1]),
                # NotChainResponse(dc.args[0], dc.args[1]),
                # NotResponse(dc.args[0], dc.args[1]),
                # NotPrecedence(dc.args[0], dc.args[1])
                           ])
        return ls
