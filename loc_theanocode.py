#from __future__ import print_function, division
import inspect

from sympy.utilities import default_sort_key
from sympy.external import import_module

from sympy.printing.printer import Printer
import sympy
from functools import partial
import sys

theano = import_module('theano')
if theano:
    ts = theano.scalar
    tt = theano.tensor
    from theano import sandbox
    from theano.sandbox import linalg as tlinalg

    mapping = {
            sympy.Add: tt.add,
            sympy.Mul: tt.mul,
            sympy.Abs: tt.abs_,
            sympy.sign: tt.sgn,
            sympy.ceiling: tt.ceil,
            sympy.floor: tt.floor,
            sympy.log: tt.log,
            sympy.exp: tt.exp,
            sympy.sqrt: tt.sqrt,
            sympy.cos: tt.cos,
            sympy.acos: tt.arccos,
            sympy.sin: tt.sin,
            sympy.asin: tt.arcsin,
            sympy.tan: tt.tan,
            sympy.atan: tt.arctan,
            sympy.atan2: tt.arctan2,
            sympy.cosh: tt.cosh,
            sympy.acosh: tt.arccosh,
            sympy.sinh: tt.sinh,
            sympy.asinh: tt.arcsinh,
            sympy.tanh: tt.tanh,
            sympy.atanh: tt.arctanh,
            sympy.re: tt.real,
            sympy.im: tt.imag,
            sympy.arg: tt.angle,
            sympy.erf: tt.erf,
            sympy.gamma: tt.gamma,
            sympy.loggamma: tt.gammaln,
            sympy.Pow: tt.pow,
            sympy.Eq: tt.eq,
            sympy.StrictGreaterThan: tt.gt,
            sympy.StrictLessThan: tt.lt,
            sympy.LessThan: tt.le,
            sympy.GreaterThan: tt.ge,
            sympy.Max: tt.maximum,  # Sympy accept >2 inputs, Theano only 2
            sympy.Min: tt.minimum,  # Sympy accept >2 inputs, Theano only 2

            # Matrices
            sympy.MatAdd: tt.Elemwise(ts.add),
            sympy.HadamardProduct: tt.Elemwise(ts.mul),
            sympy.Trace: tlinalg.trace,
            sympy.Determinant : tlinalg.det,
            sympy.Inverse: tlinalg.matrix_inverse,
            sympy.Transpose: tt.DimShuffle((False, False), [1, 0]),
    }

class TheanoPrinter(Printer):
    """ Code printer for Theano computations """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', dict())
        super(TheanoPrinter, self).__init__(*args, **kwargs)

    def _print_Symbol(self, s, dtypes={}, broadcastables={}):
        dtype = dtypes.get(s, 'floatX')
        broadcastable = broadcastables.get(s, ())
        key = (s.name, dtype, broadcastable, type(s))
        if key in self.cache:
            return self.cache[key]
        else:
            value = tt.tensor(name=s.name, dtype=dtype, broadcastable=broadcastable)
            self.cache[key] = value
            return value

    def _print_AppliedUndef(self, s, dtypes={}, broadcastables={}):
        print "------------\nApplied undefined function"
        print s
        dtype = dtypes.get(s, 'floatX')
        broadcastable = broadcastables.get(s, ())
        name = str(type(s)) + '_' + str(s.args[0])
        key = (name, dtype, broadcastable, type(s), s.args)
        print 'function key: ',key
        print 'internal function cache\n',self.cache
        print '--Done setting up AppliedUndef'
        if key in self.cache:
            print 'Key found in cache'
            return self.cache[key]
        elif hasattr(s,'__customOpID__'):
            print 'Custom Op found'
            newkey = key[0:3] + ('SymUnDef',) + key[4:]
            if newkey in self.cache:
                print 'New key found'
                kwargs = {}
                print 'arguments: ',s.args
                print 'arg types: ',[type(arg) for arg in s.args]
                children = [self._print(arg, **kwargs) for arg in s.args]
                print 'children: ',children
                print 'child types: ',[type(child) for child in children]
                return self.cache[newkey](*children)
            else:
                print 'New key not found' 
                value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
                self.cache[key] = value
                return value
        #    #Check to see if the function has been passed to the cache
        #    if str(type(s)) in self.cache.keys():
        #       print 'key',key
        #       children = [self._print(arg, **kwargs) for arg in s.args]
        #       print 'children', children
        #       return self.cache[str(type(s))](*children)
        #    else:
        #        sys.exit('Custom theano functions must be passed into the local cache')
        else:
            print 'Key not found'
            value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
            self.cache[key] = value
            return value


    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        print '--------------\nBasic example input'
        print 'expr: ',expr
        print 'op: ',op
        print 'op type: ',type(op)
        print 'kwargs: ',kwargs
        children = [self._print(arg, **kwargs) for arg in expr.args]
        print 'children: ',children
        print '--Done with Basic'
        return op(*children)

    def _print_Number(self, n, **kwargs):
        return eval(str(n))

    def _print_MatrixSymbol(self, X, dtypes={}, **kwargs):
        dtype = dtypes.get(X, 'floatX')
        # shape = [self._print(d, dtypes) for d in X.shape]
        key = (X.name, dtype, type(X))
        if key in self.cache:
            return self.cache[key]
        else:
            value = tt.Tensor(dtype, (False, False))(X.name)
            self.cache[key] = value
            return value

    def _print_DenseMatrix(self, X, **kwargs):
        return tt.stacklists([[self._print(arg, **kwargs) for arg in L]
                                     for L in X.tolist()])
    _print_ImmutableMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return tt.join(0, *[tt.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return 3.141592653589793

    def _print_Piecewise(self, expr, **kwargs):
        e, cond = expr.args[0].args
        if len(expr.args) == 1:  # TODO: What to do if one expr-cond?
            return self._print(e, **kwargs)
        return tt.switch(self._print(cond, **kwargs), self._print(e, **kwargs),
                self._print(sympy.Piecewise(*expr.args[1:]), **kwargs))

    def _print_Rational(self, expr, **kwargs):
        return tt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = tt.Rop(rv, var, tt.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        print 'Empty print: ',expr
        return expr

    def doprint(self, expr, **kwargs):
        """Returns printer's representation for expr (as a string)"""
        return self._print(expr, **kwargs)

global_cache = {}

def theano_code(expr, cache=global_cache, **kwargs):
    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims={}, broadcastables={}, keys=()):
    """ Handle various input types for dimensions in tensor_wrap

    See Also:
        tensor_wrap
        theano_funciton
    """
    if dim:
        dims = dict(zip(inputs, [dim]*len(inputs)))
    if dims:
        maxdim = max(dims.values())
        broadcastables = dict((i, (False,)*dims[i] + (True,)*(maxdim-dims[i]))
                         for i in inputs)
    return broadcastables


def local_access_test():
    return 'Local acheived'


def theano_function(inputs, outputs, dtypes={}, cache=global_cache, **kwargs):
    """ Create Theano function from SymPy expressions """
    function_arg_names = inspect.getargspec(theano.function)[0]
    if set(function_arg_names) & set(kwargs.keys()):
        theano_function_kwargs = {}
        dim_handling_kwargs = {}
        for k, v in kwargs.items():
            if k in function_arg_names:
                theano_function_kwargs[k] = v
            else:
                dim_handling_kwargs[k] = v
    else:
        theano_function_kwargs = {}
        dim_handling_kwargs = kwargs
    broadcastables = dim_handling(inputs, **dim_handling_kwargs)
    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs  = map(code, inputs)
    toutputs = map(code, outputs)
    toutputs = toutputs[0] if len(toutputs) == 1 else toutputs
    return theano.function(tinputs, toutputs, **theano_function_kwargs)
