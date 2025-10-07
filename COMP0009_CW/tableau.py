from typing import override


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

class Token:
    def __init__(self, tokenType: int, token: str) -> None:
        self.tokenType: int = tokenType
        self.token: str = token

    @override
    def __str__(self) -> str:
        return f"Token({self.token}, {self.tokenType})"

class ASTNodeType:
    ATOM: int = 0
    NEGATION: int = 1
    QUANTIFIED: int = 2
    PRED: int = 3
    CONNECTIVE: int = 4

class Quantifier:
    EXISTS: int = 0
    FORALL: int = 1


class ASTNode:
    def __init__(self, nodeType: int, fullStr: str = "") -> None:
        self.nodeType: int = nodeType
        self.fullStr: str = fullStr

class AtomNode(ASTNode):
    def __init__(self, token: int, fullStr: str) -> None:
        super().__init__(ASTNodeType.ATOM, fullStr)
        self.token: int = token

class QuantifierNode(ASTNode):
    def __init__(self, quantifer: int, fmla: ASTNode, fullStr: str) -> None:
        super().__init__(ASTNodeType.QUANTIFIED, fullStr)
        self.quantifier: int = quantifer
        self.fmla: ASTNode = fmla

class ConnectiveNode(ASTNode):
    def __init__(self, lhs: ASTNode, connective: Token, rhs: ASTNode, fullStr: str) -> None:
        super().__init__(ASTNodeType.CONNECTIVE, fullStr)
        self.lhs: ASTNode = lhs
        self.rhs: ASTNode = rhs
        self.connective: Token = connective

class NegationNode(ASTNode):
    def __init__(self, fmla: ASTNode, fullStr: str) -> None:
        super().__init__(ASTNodeType.NEGATION, fullStr)
        self.fmla: ASTNode = fmla

class PredNode(ASTNode):
    def __init__(self, args: list[ASTNode], fullStr: str) -> None:
        super().__init__(ASTNodeType.PRED, fullStr)
        self.args: list[ASTNode] = args

def tokenise(fmla: str) -> list[Token] | None: 
    tokens: list[Token] = []
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
            tokens.append(Token(Tokens.NEGATION, '~'))
        elif cur == '(':
            tokens.append(Token(Tokens.OPEN_PAREN, '('))
        elif cur == ')':
            tokens.append(Token(Tokens.CLOSE_PAREN, ')'))
        elif cur == '\\':
            if peek(idx) == '/':
                tokens.append(Token(Tokens.OR, '\\/'))
            else:
                print(f"Expected /, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == '/':
            if peek(idx) == '\\':
                tokens.append(Token(Tokens.AND, '/\\'))
            else:
                print(f"Expected \\, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == '=':
            if peek(idx) == '>':
                tokens.append(Token(Tokens.IMPLIES, '=>'))
            else:
                print(f"Expected >, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == 'E':
            tokens.append(Token(Tokens.EXIST, 'E'))
        elif cur == 'A':
            tokens.append(Token(Tokens.EXIST, 'A'))

        elif cur in preds:
            tokens.append(Token(Tokens.PRED, cur))
        elif cur in vars:
            tokens.append(Token(Tokens.VAR, cur))
        elif cur in props:
            tokens.append(Token(Tokens.PROP, cur))
        elif cur == ',':
            tokens.append(Token(Tokens.COMMA, ','))
        else:
            print(f"Unknown char{cur}")
            return None

        idx += 1

    return tokens

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.idx: int = 0
        self.tokens: list[Token] = tokens
        self.len: int = len(tokens)
        self.valid: bool = True
        self.ast: ASTNode = self.parse()
        if self.idx < self.len:
            print(f"Parsing complete, but idx at {self.idx}, less than {self.len}")
            self.valid = False
        self.isFOL: bool = False

    def peek(self) -> Token | None:
        if self.idx + 1 >= self.len:
            return None

        return self.tokens[self.idx + 1]

    def consume(self, count: int = 1) -> None:
        self.idx += count


    def match(self, token: int) -> bool:
        if self.idx >= self.len:
            print("Unexpected EOF")
            self.valid = False
            return False
        return self.tokens[self.idx].tokenType == token


    def match_peek(self, token: int) -> bool:
        peek = self.peek()
        if self.idx >= self.len or peek is None:
            print("Unexpected EOF")
            self.valid = False
            return False

        return peek.tokenType == token


    def parse(self) -> ASTNode:
        if (self.idx >= self.len):
            print("Unexpected EOF")
            self.valid = False
            return ASTNode(0)
        cur = self.tokens[self.idx]
        # print(f"idx: {self.idx} cur: {cur}")

        match (cur.tokenType):
            case Tokens.NEGATION:
                self.consume()
                res = self.parse()
                return NegationNode(res, cur.token + res.fullStr)

            case Tokens.EXIST | Tokens.FORALL:
                if not self.match_peek(Tokens.VAR):
                    print(f"Expected VAR, got {self.peek()}")
                    self.valid = False
                    return ASTNode(0)

                var = self.peek()
                assert(var is not None)

                self.isFOL = True

                self.consume(2)
                quantifier = Quantifier.EXISTS if cur.tokenType == Tokens.EXIST else Quantifier.FORALL
                res = self.parse()
                return QuantifierNode(quantifier, res, cur.token + var.token + res.fullStr)

            case Tokens.OPEN_PAREN:
                self.consume()
                lhs = self.parse()

                connective = self.tokens[self.idx]
                connectives = [Tokens.OR, Tokens.AND, Tokens.IMPLIES]

                if connective.tokenType not in connectives:
                    print(f"Expected connective, got {connective}")
                    self.valid = False
                    return ASTNode(0)

                self.consume()
                rhs = self.parse()

                if not self.match(Tokens.CLOSE_PAREN):
                    print(f"Expected CLOSE_PAREN, got {self.peek()}")
                    self.valid = False
                    return ASTNode(0)

                self.consume()

                # TODO: make this the right connectvie


                return ConnectiveNode(lhs, connective, rhs, 
                    cur.token + lhs.fullStr + connective.token + rhs.fullStr + ')'
                )

            case Tokens.PRED:
                peek = self.peek()
                if peek is None or peek.tokenType != Tokens.OPEN_PAREN:
                    self.valid = False
                    print(f"Expected OPEN_PAREN, got {self.peek()}")
                    return ASTNode(0)

                self.consume()

                s = f"{cur.token}("

                args: list[ASTNode] = []

                while True:
                    self.consume()
                    res = self.parse()
                    args.append(res)
                    s += res.fullStr
                    if self.tokens[self.idx].tokenType == Tokens.COMMA:
                        s += ','
                        continue
                    break

                if not self.match(Tokens.CLOSE_PAREN):
                    self.valid = False
                    print(f"Expected CLOSE_PAREN (PRED), got {self.tokens[self.idx]}")
                    return ASTNode(0)

                self.consume()
                s += ')'

                return PredNode(args, s)

            case Tokens.PROP | Tokens.VAR:
                self.consume()
                return AtomNode(cur.tokenType, cur.token)

            case _:
                print(f"Unexpected token: {cur}")
                self.valid = False
                return ASTNode(0)



# Parse a formula, consult parseOutputs for return values.
def parse(fmla: str):
    tokens = tokenise(fmla)

    if tokens is None:
        return ParseOutputs.NOT_FORMULA

    parser = Parser(tokens)

    if not parser.valid:
        return ParseOutputs.NOT_FORMULA

    match parser.ast.nodeType:
        case ASTNodeType.NEGATION:
            if parser.isFOL:
                return ParseOutputs.FOL_NEGATION
            return ParseOutputs.PROP_NEGATION
        case ASTNodeType.QUANTIFIED:
            assert(isinstance(parser.ast, QuantifierNode))

            if parser.ast.quantifier == Quantifier.FORALL:
                return ParseOutputs.UNI_QUANTIFIED
            return ParseOutputs.EXS_QUANTIFIED
        case ASTNodeType.CONNECTIVE:
            if parser.isFOL:
                return ParseOutputs.BIN_FOL
            return ParseOutputs.BIN_PROP

        case ASTNodeType.ATOM:
            assert(isinstance(parser.ast, AtomNode))

            if parser.ast.token == Tokens.PROP:
                return ParseOutputs.PROPOSITION

            return ParseOutputs.NOT_FORMULA
        case ASTNodeType.PRED:
            return ParseOutputs.ATOM

        case _:
            return ParseOutputs.NOT_FORMULA


# Return the LHS of a binary connective formula
def lhs(fmla: str):
    tokens = tokenise(fmla)
    assert(tokens is not None)
    parser = Parser(tokens)

    assert(isinstance(parser.ast, ConnectiveNode))

    return parser.ast.lhs.fullStr

# Return the connective symbol of a binary connective formula
def con(fmla):
    tokens = tokenise(fmla)
    assert(tokens is not None)
    parser = Parser(tokens)

    assert(isinstance(parser.ast, ConnectiveNode))

    return parser.ast.connective.token

# Return the RHS symbol of a binary connective formula
def rhs(fmla):
    tokens = tokenise(fmla)
    assert(tokens is not None)
    parser = Parser(tokens)

    assert(isinstance(parser.ast, ConnectiveNode))

    return parser.ast.rhs.fullStr


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
