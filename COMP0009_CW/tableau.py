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

    def copy(self):
        return Token(self.tokenType, self.token)

class ASTNodeType:
    QUANTIFIED: int = 0
    PRED: int = 1
    CONNECTIVE: int = 2
    NEGATION: int = 3
    ATOM: int = 4

class Quantifier:
    EXISTS: int = 0
    FORALL: int = 1


class ASTNode:
    def __init__(self, nodeType: int, fullStr: str = "") -> None:
        self.nodeType: int = nodeType
        self.fullStr: str = fullStr

    @override
    def __eq__(self, value) -> bool:
        return self.fullStr == value.fullStr

    @override
    def __str__(self) -> str:
        return self.fullStr

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def copy(self):
        return ASTNode(self.nodeType, self.fullStr)

class AtomNode(ASTNode):
    def __init__(self, token: int, fullStr: str) -> None:
        super().__init__(ASTNodeType.ATOM, fullStr)
        self.token: int = token

    @override
    def copy(self):
        return AtomNode(self.token, self.fullStr)

    @override
    def __eq__(self, other: ASTNode):
        if not isinstance(other, AtomNode):
            return False
        
        return self.token == other.token and self.fullStr == other.fullStr

class QuantifierNode(ASTNode):
    def __init__(self, quantifer: int, fmla: ASTNode, var: str, fullStr: str) -> None:
        super().__init__(ASTNodeType.QUANTIFIED, fullStr)
        self.quantifier: int = quantifer
        self.var = var
        self.fmla: ASTNode = fmla
        self.usedConstants: list[str] = []

    @override
    def copy(self):
        return QuantifierNode(self.quantifier, self.fmla.copy(), self.var, self.fullStr)

    @override
    def __eq__(self, other: ASTNode):
        if not isinstance(other, QuantifierNode):
            return False
        return self.quantifier == other.quantifier and self.var == other.var and self.fmla == other.fmla

    @override
    def __str__(self):
        quantifier = "E" if self.quantifier == Quantifier.EXISTS else "A"
        return quantifier + self.var + "<" + self.fmla.__str__() + ">"

class ConnectiveNode(ASTNode):
    def __init__(self, lhs: ASTNode, connective: Token, rhs: ASTNode, fullStr: str) -> None:
        super().__init__(ASTNodeType.CONNECTIVE, fullStr)
        self.lhs: ASTNode = lhs
        self.rhs: ASTNode = rhs
        self.connective: Token = connective

    @override
    def copy(self):
        return ConnectiveNode(self.lhs.copy(), self.connective.copy(), self.rhs.copy(), self.fullStr)

    @override
    def __eq__(self, other: ASTNode):
        if not isinstance(other, ConnectiveNode):
            return False

        return self.lhs == other.lhs and self.connective == other.connective and self.rhs == other.rhs

    @override
    def __str__(self) -> str:
        return "(" + self.lhs.__str__() + self.connective.token + self.rhs.__str__() + ")"

class NegationNode(ASTNode):
    def __init__(self, fmla: ASTNode, fullStr: str) -> None:
        super().__init__(ASTNodeType.NEGATION, fullStr)
        self.fmla: ASTNode = fmla
        
    @override
    def copy(self):
        return NegationNode(self.fmla.copy(), self.fullStr)

    @override
    def __eq__(self, other: ASTNode):
        if not isinstance(other, NegationNode):
            return False

        return self.fmla == other.fmla

    @override
    def __str__(self) -> str:
        return "~<" + self.fmla.__str__() + ">"

class PredNode(ASTNode):
    def __init__(self, args: list[ASTNode], fnSymbol: str, fullStr: str) -> None:
        super().__init__(ASTNodeType.PRED, fullStr)
        self.args: list[ASTNode] = args
        self.fnSymbol = fnSymbol

    @override
    def copy(self):
        return PredNode([a.copy() for a in self.args], self.fnSymbol, self.fullStr)

    @override
    def __eq__(self, other: ASTNode):
        if not isinstance(other, PredNode):
            return False

        return self.args == other.args

    @override
    def __str__(self) -> str:
        args = ""
        for arg in self.args:
            args += f"{arg},"
        return self.fnSymbol + "(" + args + ") "

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
        elif cur == '&':
            tokens.append(Token(Tokens.AND, "&"))
        elif cur == '-':
            if peek(idx) == '>':
                tokens.append(Token(Tokens.IMPLIES, '->'))
            else:
                print(f"Expected >, got {cur} at pos {idx}")
                return None

            idx += 1
        elif cur == 'E':
            tokens.append(Token(Tokens.EXIST, 'E'))
        elif cur == 'A':
            tokens.append(Token(Tokens.FORALL, 'A'))

        elif cur in preds:
            tokens.append(Token(Tokens.PRED, cur))
        elif cur in vars:
            tokens.append(Token(Tokens.VAR, cur))
        elif cur in props:
            tokens.append(Token(Tokens.PROP, cur))
        elif cur == ',':
            tokens.append(Token(Tokens.COMMA, ','))
        else:
            print(f"Unexpected char{cur}")
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
                return QuantifierNode(quantifier, res, var.token, cur.token + var.token + res.fullStr)

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

                return ConnectiveNode(lhs, connective, rhs, 
                    cur.token + lhs.fullStr + connective.token + rhs.fullStr + ')'
                )

            case Tokens.PRED:
                peek = self.peek()
                if peek is None or peek.tokenType != Tokens.OPEN_PAREN:
                    self.valid = False
                    print(f"Expected OPEN_PAREN, got {peek}")
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

                return PredNode(args, cur.token[0],s)

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


class TablaeuTree:
    def __init__(self, node: ASTNode | None) -> None:
        self.nodes: list[ASTNode] = [] # Maybe use a tablaeu node? but we don't need to store if it's checked - simply not push
        self.posLiterals: list[ASTNode] = []
        self.negLiterals: list[ASTNode] = []
        self.seenAtoms: list[str] = []
        self.nextConstant: str = "c"
        self.dormantGammas: list[QuantifierNode] = []

        if node is not None:
            self.push(node)

    def next_expansion(self):
        highestPriority = self.nodes[0].nodeType
        highestIdx = 0

        for i in range(len(self.nodes)):
            cur = self.nodes[i]
            if cur.nodeType > highestPriority:
                highestPriority = cur.nodeType
                highestIdx = i

        return self.nodes[highestIdx], highestIdx

    def activate_dormants(self):
        for node in self.dormantGammas: # Reactivate gammas when expansions are needed again
            self.nodes.append(node)

    def mark(self, idx):
        _ = self.nodes.pop(idx)

    def expanded(self) -> bool:
        return len(self.nodes) == 0

    def new_constant(self) -> str:
        ret = self.nextConstant
        self.nextConstant += "c"
        self.seenAtoms.append(ret)
        self.activate_dormants()
        return ret

    def any_constant(self, usedConstants: list[str]):
        for i in self.seenAtoms:
            if i not in usedConstants:
                return i
        return ""

    def contradiction(self) -> bool:
        for node in self.posLiterals:
            for nNode in self.negLiterals:
                if node == nNode:
                    return True
        return False

    def push(self, node: ASTNode):
        # Check if it's an atom
        workingNode = node
        isNegation = False
        if isinstance(node, NegationNode):
            workingNode = node.fmla
            isNegation = True

        if workingNode.nodeType == ASTNodeType.ATOM or workingNode.nodeType == ASTNodeType.PRED:
            if isNegation:
                self.negLiterals.append(workingNode)
            else:
                self.posLiterals.append(workingNode)
        else:
            self.nodes.append(node)

        # print(f"Pushing: {node}")

        # print("Current queue:")
        # print(self.nodes)
        # print(self.negLiterals)
        # print(self.posLiterals)

    def clone(self):
        ret = TablaeuTree(None)
        ret.nodes = [a.copy() for a in self.nodes]
        ret.posLiterals = [a.copy() for a in self.posLiterals]
        ret.negLiterals = [a.copy() for a in self.negLiterals]

        return ret

# You may choose to represent a theory as a set or a list
def theory(fmla: str):#initialise a theory with a single formula in it
    tokens = tokenise(fmla)
    assert(tokens is not None)
    return TablaeuTree(Parser(tokens).ast)


def replace(original: str, to: str, node: ASTNode):
    if isinstance(node, AtomNode) and node.fullStr == original:
        if node.token != Tokens.VAR:
            print("wtf")
        node.fullStr = to

    elif isinstance(node, PredNode):
        for arg in node.args:
            replace(original, to, arg)

    elif isinstance(node, ConnectiveNode):
        replace(original, to, node.lhs)
        replace(original, to, node.rhs)

    elif isinstance(node, NegationNode):
        replace(original, to, node.fmla)

    elif isinstance(node, QuantifierNode):
        if node.var == original:
            print(f"Replacing {original} with {to}, but variable reencounted as quantifier {node}?")
        else:
            replace(original, to, node.fmla)


def is_alpha(node: ASTNode) -> tuple[bool, list[ASTNode]]:
    if isinstance(node, ConnectiveNode) and node.connective.tokenType == Tokens.AND:
        return True, [node.lhs, node.rhs]

    if isinstance(node, NegationNode):
        inner = node.fmla
        if isinstance(inner, ConnectiveNode):
            if inner.connective.tokenType == Tokens.OR:
                return True, [NegationNode(inner.lhs, "~" + inner.rhs.fullStr), NegationNode(inner.rhs, "~" + inner.rhs.fullStr)]
            elif inner.connective.tokenType == Tokens.IMPLIES:
                return True, [inner.lhs, NegationNode(inner.rhs, "~" + inner.rhs.fullStr)]

        elif isinstance(inner, NegationNode):
            return True, [inner.fmla]

    return False, []

def is_beta(node: ASTNode) -> tuple[bool, list[ASTNode]]:
    if isinstance(node, ConnectiveNode):
        if node.connective.tokenType == Tokens.OR:
            return True, [node.lhs, node.rhs]
        elif node.connective.tokenType == Tokens.IMPLIES:
            return True, [NegationNode(node.lhs, "~" + node.lhs.fullStr)]

    elif isinstance(node, NegationNode):
        inner = node.fmla
        if isinstance(inner, ConnectiveNode) and inner.connective.tokenType == Tokens.AND:
            return True, [NegationNode(inner.lhs, "~" + inner.rhs.fullStr), NegationNode(inner.rhs, "~" + inner.rhs.fullStr)]

    return False, []

def is_gamma(node: ASTNode) -> tuple[bool, QuantifierNode, ASTNode]:
    if isinstance(node, QuantifierNode):
        if node.quantifier == Quantifier.FORALL:
            return True, node, node.fmla

    if isinstance(node, NegationNode):
        inner = node.fmla
        if isinstance(inner, QuantifierNode) and inner.quantifier == Quantifier.EXISTS:
            return True, inner, NegationNode(inner.fmla, inner.fmla.fullStr)

    return False, QuantifierNode(0, ASTNode(0, ""), "", ""), ASTNode(0)

def is_delta(node: ASTNode) -> tuple[bool, str, ASTNode]:
    # existential
    if isinstance(node, QuantifierNode):
        if node.quantifier == Quantifier.EXISTS:
            return True, node.var, node.fmla

    if isinstance(node, NegationNode):
        inner = node.fmla
        if isinstance(inner, QuantifierNode) and inner.quantifier == Quantifier.FORALL:
            return True, inner.var, NegationNode(inner.fmla, inner.fmla.fullStr)

    return False, "", ASTNode(0)

#check for satisfiability
def sat(tableau: list[TablaeuTree]):
#output 0 if not satisfiable, output 1 if satisfiable, output 2 if number of constants exceeds MAX_CONSTANT
    iter = 0
    while len(tableau):
        branch = tableau.pop(0)

        # if (iter > 20):
        #     print("max recur limt")
        #     break
        # else:
        #     iter += 1

        if len(branch.nextConstant) > MAX_CONSTANTS:
            return 2

        if not branch.contradiction() and branch.expanded():
            return 1

        node, idx = branch.next_expansion()

        # print(f"Expanding node {node}")

        # alpha
        isAlpha, fmlas = is_alpha(node)
        if isAlpha:
            branch.mark(idx)

            for fmla in fmlas:
                branch.push(fmla)

            if not branch.contradiction():
                tableau.append(branch)

        isBeta, fmlas = is_beta(node)
        if isBeta:
            for fmla in fmlas:
                working = branch.clone()
                working.mark(idx)
                working.push(fmla)

                if not working.contradiction():
                    tableau.append(working)

        isGamma, quantifier, fmla = is_gamma(node)
        if isGamma:
            replaceFrom, replaceTo = quantifier.var, branch.any_constant(quantifier.usedConstants)

            if replaceTo == "":
                branch.mark(idx)
                branch.dormantGammas.append(quantifier)
                if not branch.contradiction():
                    tableau.append(branch)
            else:
                # print("Expanding with constant ", replaceTo)
                quantifier.usedConstants.append(replaceTo)

                fmlaCopy = fmla.copy()

                replace(replaceFrom, replaceTo, fmlaCopy)
                branch.push(fmlaCopy)

                if not branch.contradiction():
                    tableau.append(branch)

        isDelta, var, fmla = is_delta(node)
        if isDelta:
            replaceFrom, replaceTo = var, branch.new_constant()

            replace(replaceFrom, replaceTo, fmla)
            
            branch.mark(idx)
            branch.push(fmla)

            if not branch.contradiction():
                tableau.append(branch)

        if not isDelta and not isGamma and not isBeta and not isAlpha:
            raise RuntimeError("?")

        # TODO: all the different types of expanding

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
