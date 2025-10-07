
import token


MAX_CONSTANTS = 10

class ParseOutputs:
    NOT_FORMULA: int = 0
    ATOM: int = 1
    FOL_NEGATION: int = 2
    UNI_QUANTIFIED: int = 3
    EXS_QUANTIFIED: int = 4
    BIN_FOL: int = 5
    PROPOSITION: int = 6
    PROP_NEGATION: int = 7
    BIN_PROP: int = 8

class Tokens:
    PROP: int = 0
    VAR: int = 1
    OPEN_PAREN: int = 2
    CLOSE_PAREN: int = 3
    NEGATION: int = 4
    AND: int = 5
    OR: int = 6
    IMPLIES: int = 7
    EXIST: int = 8
    FORALL: int = 9
    PRED: int = 10
    COMMA: int = 11

def tokenise(fmla: str) -> list[int] | None: 
    # print(f"tokenising {fmla}")

    tokens: list[int] = []
    vars = "wxyz"
    props = "pqrs"
    preds = "PQRS"

    def peek(idx: int) -> str | None:
        if (idx + 1 >= len(fmla)):
            return None
        
        return fmla[idx + 1]

    idx = 0

    while idx < len(fmla):
        cur = fmla[idx]
        if cur == '~':
            tokens.append(Tokens.NEGATION)
        elif cur == '(':
            tokens.append(Tokens.OPEN_PAREN)
        elif cur == ')':
            tokens.append(Tokens.CLOSE_PAREN)
        elif cur == '\\':
            if peek(idx) == '/':
                tokens.append(Tokens.OR)
            else:
                print(f"Expected /, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == '/':
            if peek(idx) == '\\':
                tokens.append(Tokens.AND)
            else:
                print(f"Expected \\, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == '=':
            if peek(idx) == '>':
                tokens.append(Tokens.IMPLIES)
            else:
                print(f"Expected >, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == 'E':
            tokens.append(Tokens.EXIST)
        elif cur == 'A':
            tokens.append(Tokens.FORALL)

        elif cur in preds:
            tokens.append(Tokens.PRED)
        elif cur in vars:
            tokens.append(Tokens.VAR)
        elif cur in props:
            tokens.append(Tokens.PROP)
        elif cur == ',':
            tokens.append(Tokens.COMMA)
        else:
            print(f"Unknown char{cur}")
            return None

        idx += 1

    return tokens



# Parse a formula, consult parseOutputs for return values.
def parse(fmla: str):
    tokens = tokenise(fmla)

    if tokens is None:
        return ParseOutputs.NOT_FORMULA

    # Parse the tokens
    # print(tokens)

    stack: list[int] = [] # Keep track of seen tokens

    # Helpers
    def peek(idx: int) -> int | None:
        if (idx + 1 >= len(fmla)):
            return None
        
        return tokens[idx + 1]

    def last() -> int | None:
        if len(stack) == 0:
            return None
        return stack[len(stack) - 1]

    fol = False

    idx = 0
    while idx < len(tokens):
        cur = tokens[idx]
        # print(stack)

        if cur == Tokens.OPEN_PAREN:
            stack.append(cur)
        elif cur == Tokens.CLOSE_PAREN: # Signals end of either a parenthesis (a binary connective) or a predicate
            if last() == Tokens.PRED:
                _ = stack.pop()
            elif last() == Tokens.AND or last() == Tokens.OR or last() == Tokens.IMPLIES:
                _ = stack.pop()
                _ = stack.pop()
            else:
                print(f"Unexpected close parenthesis?")
                return ParseOutputs.NOT_FORMULA

        elif cur == Tokens.PRED:
            stack.append(cur)
            if peek(idx) != Tokens.OPEN_PAREN:
                print(f"Expected OPEN_PAREN, got {cur}")
                return ParseOutputs.NOT_FORMULA

            idx += 1

        elif cur == Tokens.EXIST or cur == Tokens.FORALL:
            fol = True

            if peek(idx) != Tokens.VAR:
                print(f"Expected VAR, got {cur}")
                return ParseOutputs.NOT_FORMULA

            idx += 1 # Just parse for formula

        elif cur == Tokens.OR:
            if last() != Tokens.OPEN_PAREN:
                print(f"Unexpected OR... Missing open parenthesis?")
                return ParseOutputs.NOT_FORMULA

            stack.append(cur)

        elif cur == Tokens.AND:
            if last() != Tokens.OPEN_PAREN:
                print(f"Unexpected AND... Missing open parenthesis?")
                return ParseOutputs.NOT_FORMULA

            stack.append(cur)

        elif cur == Tokens.IMPLIES:
            if last() != Tokens.OPEN_PAREN:
                print(f"Unexpected IMPLIES... Missing open parenthesis?")
                return ParseOutputs.NOT_FORMULA

            stack.append(cur)

        idx += 1

    if len(stack) > 0:
        print("Unexpected EOF?")
        return ParseOutputs.NOT_FORMULA
    if len(tokens) == 1:
        return ParseOutputs.ATOM
    if tokens[0] == Tokens.FORALL:
        return ParseOutputs.UNI_QUANTIFIED
    elif tokens[0] == Tokens.EXIST:
        return ParseOutputs.EXS_QUANTIFIED


    return ParseOutputs.NOT_FORMULA

# Return the LHS of a binary connective formula
def lhs(fmla: str):
    parenCount: int = 0

    for idx in range(len(fmla)):
        if fmla[idx] == '(':
            parenCount += 1
        elif fmla[idx] == ')':
            parenCount -= 1





    return ''

# Return the connective symbol of a binary connective formula
def con(fmla):
    return ''

# Return the RHS symbol of a binary connective formula
def rhs(fmla):
    return ''


# You may choose to represent a theory as a set or a list
def theory(fmla):#initialise a theory with a single formula in it
    return None

#check for satisfiability
def sat(tableau):
#output 0 if not satisfiable, output 1 if satisfiable, output 2 if number of constants exceeds MAX_CONSTANTS
    return 0

#------------------------------------------------------------------------------------------------------------------------------:
#                   DO NOT MODIFY THE CODE BELOW. MODIFICATION OF THE CODE BELOW WILL RESULT IN A MARK OF 0!                   :
#------------------------------------------------------------------------------------------------------------------------------:

f = open('input.txt')

parseOutputs = ['not a formula',
                'an atom',
                'a negation of a first order logic formula',
                'a universally quantified formula',
                'an existentially quantified formula',
                'a binary connective first order formula',
                'a proposition',
                'a negation of a propositional formula',
                'a binary connective propositional formula']

satOutput = ['is not satisfiable', 'is satisfiable', 'may or may not be satisfiable']



firstline = f.readline()

PARSE = False
if 'PARSE' in firstline:
    PARSE = True

SAT = False
if 'SAT' in firstline:
    SAT = True

for line in f:
    if line[-1] == '\n':
        line = line[:-1]
    parsed = parse(line)

    if PARSE:
        output = "%s is %s." % (line, parseOutputs[parsed])
        if parsed in [5,8]:
            output += " Its left hand side is %s, its connective is %s, and its right hand side is %s." % (lhs(line), con(line) ,rhs(line))
        print(output)

    if SAT:
        if parsed:
            tableau = [theory(line)]
            print('%s %s.' % (line, satOutput[sat(tableau)]))
        else:
            print('%s is not a formula.' % line)
