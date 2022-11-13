from dataloading.Log import TracePositional
from embeddings.declare.Utils import DeclareUnary, DeclareBinary, DeclareCandidate

class Absence(DeclareUnary):
    def __init__(self, n, arg1=""):
        super().__init__("Absence", n, arg1)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count > self.n:
            return -(count - self.n)
        else:
            return 1


class Exists(DeclareUnary):
    def __init__(self, n=1, arg1=""):
        super().__init__("Exists", n, arg1)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count >= self.n:
            return count
        else:
            return -1


class Exactly(DeclareUnary):
    def __init__(self, n, arg1=""):
        super().__init__("Exactly", n, arg1)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        count = trace.eventCount(self.arg1)
        if count == self.n:
            return 1
        else:
            return -1


class Init(DeclareUnary):
    def __init__(self, arg1=""):
        super().__init__("Init", 1, arg1)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.length == 0) or (not trace.hasPos(self.arg1, 0)):
            return -1
        else:
            return 1


class ExclChoice(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ExclChoice", arg1, arg2)
        self.ex1 = Exists(1, arg1)
        self.ex2 = Exists(1, arg2)

    def __call__(self, trace):
        a = self.ex1(trace)
        b = self.ex2(trace)
        if (a == -1):
            if (b == -1):
                return -1
            else:
                return b
        else:
            if (b == -1):
                return a
            else:
                return -1


class Choice(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Choice", arg1, arg2)
        self.ex1 = Exists(1, arg1)
        self.ex2 = Exists(1, arg2)

    def __call__(self, trace):
        a = self.ex1(trace)
        b = self.ex2(trace)
        if (a == -1) and (b == -1):
            return -1
        else:
            if a == -1:
                a = 0
            if b == -1:
                b = 0
            return a + b


class AltPrecedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltPrecedence", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
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
                    return 0

                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

                # Go through all event 2's, check that there is respective event 1.
                # Find largest event 1 position, which is smaller than event 2 position

                # implementation
                # Check 1-forward, the 1-forward has to be greater than event 2 and current one has to be smaller than event2

                event_1_ind = 0
                for i, pos2 in enumerate(event_2_positions):
                    # find first in event_2_positions, it has to be before next in event_1_positions

                    while True:
                        if event_1_ind >= len(event_1_positions):
                            # out of preceding events, but there are still event 2's remaining.
                            return -1

                        next_event_1_pos = None

                        if event_1_ind < len(event_1_positions) - 1:
                            next_event_1_pos = event_1_positions[event_1_ind + 1]

                        event_1_pos = event_1_positions[event_1_ind]

                        if next_event_1_pos:
                            if event_1_pos < pos2 and next_event_1_pos > pos2:
                                # found the largest preceding event
                                event_1_ind += 1
                                break
                            elif event_1_pos > pos2 and next_event_1_pos > pos2:
                                # no event larger
                                return -1
                            else:
                                event_1_ind += 1


                        else:
                            # if no next event, check if current is smaller
                            if event_1_pos < pos2:
                                event_1_ind += 1
                                break
                            else:
                                return -1  # since there is no smaller remaining event

                count = len(event_2_positions)
                return count
            else:
                # impossible because there has to be at least one event1 with event2
                return -1
        return 0  # todo: vacuity condition!!


class AltResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltResponse", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):

                event_2_ind = 0

                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

                for i, pos1 in enumerate(event_1_positions):
                    # find first in event_2_positions, it has to be before next in event_1_positions
                    next_event_1_pos = None
                    if i < len(event_1_positions) - 1:
                        next_event_1_pos = event_1_positions[i + 1]

                    while True:
                        if event_2_ind >= len(event_2_positions):
                            # out of response events
                            return -1

                        if event_2_positions[event_2_ind] > pos1:
                            # found first greater than event 1 pos
                            # check if it is smaller than next event 1
                            if next_event_1_pos and event_2_positions[event_2_ind] > next_event_1_pos:
                                # next event 2 is after next event 1..
                                return -1
                            else:
                                # consume event 2 and break out to next event 1
                                event_2_ind += 1
                                break

                        event_2_ind += 1

                count = len(event_1_positions)
                return count
                # every event 2 position has to be after respective event 1 position and before next event 2 position
            else:
                return -1

        # Vacuously
        return 0


class AltSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("AltSuccession", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.hasEvent(self.arg1)) != (trace.hasEvent(self.arg2)):
            return -1

        if trace.hasEvent(self.arg1) and trace.hasEvent(self.arg2):
            event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

            if len(event_1_positions) != len(event_2_positions):
                return -1  # impossible if not same length

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
                    return -1  # next one is smaller than current

                pos = next_pos  # go to next one.
                switch = not switch  # swap array

            count = len(event_1_positions)
            return count

        return 0  # vacuity condition


class ChainPrecedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainPrecedence", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg2):
            if trace.hasEvent(self.arg1):
                # Each event1 must instantly be followed by event2
                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

                if len(event_1_positions) < len(event_2_positions):
                    return -1  # impossible to fulfill

                event_1_ind = 0

                for i, pos2 in enumerate(event_2_positions):
                    # find first event 2 which is after each event 1
                    while True:
                        if event_1_ind >= len(event_1_positions):
                            return -1  # not enough response

                        if pos2 < event_1_positions[event_1_ind]:
                            return -1  # passed, no event before pos2

                        if pos2 - 1 == event_1_positions[event_1_ind]:
                            event_1_ind += 1
                            break  # found right one! Move to next B event

                        event_1_ind += 1

                count = len(event_2_positions)
                return count
            else:
                return -1  # no response for event1
        return 0  # todo, vacuity


class ChainResponse(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainResponse", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                # Each event1 must instantly be followed by event2
                event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
                event_2_positions = trace.getEventsInPositionalTrace(self.arg2)

                if len(event_1_positions) > len(event_2_positions):
                    return -1  # impossible to fulfill

                event_2_ind = 0

                for i, pos1 in enumerate(event_1_positions):
                    # find first event 2 which is after each event 1
                    while True:
                        if event_2_ind >= len(event_2_positions):
                            return -1  # not enough response

                        if pos1 < event_2_positions[event_2_ind]:
                            if pos1 + 1 != event_2_positions[event_2_ind]:
                                return -1  # next one is not straight after
                            else:
                                event_2_ind += 1
                                break  # next one is straight after move to next event1
                        event_2_ind += 1

                count = len(event_1_positions)
                return count

            else:
                return -1  # no response for event1

        return 0  # todo, vacuity


class ChainSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainSuccession", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if (trace.hasEvent(self.arg1)) != (trace.hasEvent(self.arg2)):
            return -1

        if (trace.hasEvent(self.arg1)) and (trace.hasEvent(self.arg2)):
            event_1_positions = trace.getEventsInPositionalTrace(self.arg1)
            event_2_positions = trace.getEventsInPositionalTrace(self.arg2)
            if len(event_1_positions) != len(event_2_positions):
                # has to be same number of events
                return -1

            # They have to appear together, with event1 always before event2
            for i in range(len(event_1_positions)):
                if event_1_positions[i] + 1 != event_2_positions[i]:
                    return -1

            count = len(event_1_positions)
            return count

        return 0  # todo vacuity


class Succession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Succession", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if self.arg1 in trace and not self.arg2 in trace:
            return -1

        if self.arg2 in trace and not self.arg1 in trace:
            return -1

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
                return min(trace.eventCount(self.arg1), trace.eventCount(self.arg2))
            else:
                return -1

        # todo: vacuity condition!
        return 0


class RespExistence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("RespExistence", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                return min(trace.eventCount(self.arg1), trace.eventCount(self.arg2))
            else:
                return -1
        return 0  # 0, if vacuity condition


class Response(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Response", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):
                last_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[-1]
                last_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[-1]
                if last_pos_event_2 > last_pos_event_1:
                    return min(trace.eventCount(self.arg1), trace.eventCount(self.arg2))
                else:
                    # last event2 is before event1
                    return -1
            else:
                # impossible for event 2 to be after event 1 if there is no event 2
                return -1

        return 0  # not vacuity atm..

class Precedence(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("Precedence", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg2):
            if trace.hasEvent(self.arg1):
                first_pos_event_1 = trace.getEventsInPositionalTrace(self.arg1)[0]
                first_pos_event_2 = trace.getEventsInPositionalTrace(self.arg2)[0]
                if first_pos_event_1 < first_pos_event_2:
                    # todo: check frequency condition
                    return min(trace.eventCount(self.arg1), trace.eventCount(self.arg2))
                else:
                    # first position of event 2 is before first event 1
                    return -1

            else:
                # impossible because there has to be at least one event1 with event2
                return -1

        # Vacuously fulfilled
        return 0

class NotSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("NotSuccession", arg1, arg2)

    def __call__(self, trace):
        assert isinstance(trace, TracePositional)
        if trace.hasEvent(self.arg1):
            if trace.hasEvent(self.arg2):

                # for this to be true, last event 2 has to be before first event 1
                first_event_1 =  trace.getEventsInPositionalTrace(self.arg1)[0]
                last_event_2 =  trace.getEventsInPositionalTrace(self.arg2)[-1]

                if first_event_1 < last_event_2:
                    return -1  # in this case there is an event 2, which appears after first event
                else:
                    return 1
            else:
                return 1  # not possible

        # if not, then impossible and template fulfilled
        return 0  # vacuity

class NotChainSuccession(DeclareBinary):
    def __init__(self, arg1="", arg2=""):
        super().__init__("ChainSuccession", arg1, arg2)

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
                    return 1  # no more choices

                current_e1 = event_1_positions[e1_ind]
                current_e2 = event_2_positions[e2_ind]

                if current_e1 > current_e2:
                    e2_ind += 1
                else:
                    if current_e1 + 1 == current_e2:
                        return -1  # found a place, where they are together
                    e1_ind += 1

        # How to do vacuity here? 1 by default most likely
        return 0  # TODO, this condition?

class DatalessDeclare:
    def __init__(self):
        self.unary_template_map = [
            Init(),
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
             Precedence()
        ]


    def instantiateDeclares(self, candidates):
        ls = []
        for dc in candidates:
            assert isinstance(dc, DeclareCandidate)
            if len(dc.args)==1:
                ls.extend([Init(dc.args[0]),
             Exists(1,dc.args[0]),
             Absence(1,dc.args[0]),
             Absence(2,dc.args[0]),
             Absence(3,dc.args[0]),
             Exactly(1,dc.args[0]),
             Exactly(2,dc.args[0]),
             Exactly(3,dc.args[0])])
            elif len(dc.args)==2:
                ls.extend([ExclChoice(dc.args[0], dc.args[1]),
                           Choice(dc.args[0], dc.args[1]),
                           AltPrecedence(dc.args[0], dc.args[1]),
                           AltSuccession(dc.args[0], dc.args[1]),
                           AltResponse(dc.args[0], dc.args[1]),
                           ChainPrecedence(dc.args[0], dc.args[1]),
                           ChainResponse(dc.args[0], dc.args[1]),
                           ChainSuccession(dc.args[0], dc.args[1]),
                           NotChainSuccession(dc.args[0], dc.args[1]),
                           NotSuccession(dc.args[0], dc.args[1]),
                           RespExistence(dc.args[0], dc.args[1]),
                           Response(dc.args[0], dc.args[1]),
                           Succession(dc.args[0], dc.args[1]),
                           Precedence(dc.args[0], dc.args[1])])
        return ls
