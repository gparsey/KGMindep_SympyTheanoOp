import sympy
import numpy
import theano
from scipy.interpolate import splrep, splev
from loc_theanocode import *

#===============================================================================
# FUNCTION: IUF (Implemented Undefined Function)
#===============================================================================

def IUF(name, num_fun):
  """
  Create an Implemented Undefined Function.
  
  Parameters
  ----------
  name : string
    Name to be given to the symbolic function
  num_fun : callable
    Numerical implementation to be "attached" to the symbolic function
  
  Returns
  -------
  f : sympy.core.function.UndefinedFunction
    Symbolic function with an attached numerical implementation
    (automatically recognized by sympy.lambdify)
  
  Notes
  -----
  We would like to define the number of independent variables, and check if a
  wrong number of arguments is passed to the function.
  
  """
  f = sympy.Function(name) # undefined function (symbolic)
  f._imp_ = staticmethod(num_fun)
  f.__customOpID__ = None
  return f

#===============================================================================
# FUNCTION: SymRateCoeff
#===============================================================================

def SymRateCoeff (name, num_fun, num_dfun):
  """
  Create a symbolic "rate coefficient" function, with "attached" numerical
  callable functions K(T) and dK(T)/dT.
  
  Parameters
  ----------
  name : string
    Name to be given to the rate coefficient
  num_fun : callable
    Numerical implementation of the rate coefficient
  num_dfun : callable
    Numerical implementation of the 1st derivative of the rate coefficient
    
  Returns
  -------
  K : sympy.core.function.UndefinedFunction
    Symbolic rate coefficient, one time differentiable, with attached numerical
    implementation for the function itself and for its derivative.
  
  """
  K = IUF(name, num_fun)
  K.fdiff = lambda self, nvar=1 : \
                           IUF(name+'D'+str(self.args[nvar-1]),num_dfun)(*self.args)
  return K

#===============================================================================
# CLASS: spline1D
#===============================================================================

class spline1D (object):
  """
  Very simple wrapper class for creating 1D interpolating B-splines.
  This class greatly simplifies the procedure for getting the value of
  the spline or its first derivative at a given point, or set of points.
  
  This class is based on the non object-oriented wrapping of FITPACK:
    * scipy.interpolate.splrep(...)
    * scipy.interpolate.splev (...)
  
  Piecewise cubic interpolation is used if not otherwise specified.
  
  Parameters
  ----------
  x,y : float arrays
    Abscissae and ordinates of data points
  k : int
    Order of interpolating spline (must be > 0)
  
  Returns
  -------
  out : spline1D object
    Interpolating spline, with internal representation (t,c,k)  
  
  """
  # ----------------------------------------------------------------------------
  # Private function: constructor
  
  def __init__(self,x,y,k=3):
    self._tck = splrep (x,y,k=k,s=0)
  
  # ----------------------------------------------------------------------------
  # Read-only access to object attributes
  
  @property
  def tck(self):
    """ (t,c,k) tuple containing the vector of knots, the B-spline coefficients,
    and the degree of the spline."""
    return self._tck
  
  @property
  def k(self):
    """ Degree of the interpolating spline."""
    return self._tck[2]
  
  # ----------------------------------------------------------------------------
  # Functions for evaluating the spline and its 1st derivative
  
  def __call__(self,x):
    """ Return value of spline at given point (or array of points)."""
    return splev (x,self._tck,der=0)
  
  def deriv (self,x):
    """ Return value of first derivative at given point (or array of points)."""
    return splev (x,self._tck,der=1)

#===============================================================================
# CLASS: TestInterOp
#===============================================================================

class TestInterpOp:
   def __init__(self,data=None,fname='interp'):
      if data == None:
         self.xset = numpy.linspace(-100.0,100.0,1000)
         self.yset = self.xset**2
      elif len(data) == 2:
         self.xset = data[0]
         self.yset = data[1]
      else:
         sys.exit('input data is a list or tuple with two lists of data')
      self.fname = fname

   def CreateSuite(self,):
      x,y = sympy.symbols('x,y')
      pwfunc = sympy.Piecewise((y**3, y >= 0.0),((-y)**3, y< 0.0))
      sK,K,Kp = self.SymInterp()
      expr1 = x*sympy.exp(y)
      expr2 = x*pwfunc
      expr3 = x*K(y)
      self.inp = (x,y)
      self.suite = [self.inp,expr1,expr2,expr3]
      self.func = [pwfunc,K]

   def CreateFunc(self,inputs,expressions,s=False,t=False,dimset=0,typeset='float64'):
      return_list = []
      if s:
         return_list.append(sympy.lambdify(inputs,expressions,'numpy'))
      if t:
         if not hasattr(expressions,'__iter__'):
            print 'encasing function'
            exprlist = [expressions]
         else:
            exprlist = expressions
         if not hasattr(inputs,'__iter__'):
            print 'encasing inputs'
            inplist = [inputs]
         else:
            inplist = inputs
         dimdict = {}
         for i in inplist:
            dimdict[i] = dimset
         typedict = {}
         for i in inplist:
            typedict[i] = typeset
         cusOps = {}
         dtype = 'floatX'
         broadcastable = ()
         name = 'sym'+self.fname + '_' + str(self.inp[1])
         key = (name, dtype, broadcastable, 'SymUnDef', (self.inp[1],))
         print 'key: ',key
         cusOps[key] = self.TheanoInterpOp()
         print 'Custom Operations'
         for k,v in cusOps.items():
            print k, ' -> ',v
         return_list.append(theano_function(inplist,exprlist,dims=dimdict,dtypes = typedict,cache=cusOps,on_unused_input='ignore'))
         #return_list.append(sympy.printing.theanocode.theano_function(inplist,exprlist,dims=dimdict,dtypes = typedict,cache=cusOps,on_unused_input='ignore'))
      return return_list

   def SymInterp(self,):
      return s1d_output(self.yset,self.xset,'sym'+self.fname)

   def TheanoInterpOp(self,):
      return TheanoInterpWrapOp('theano'+self.fname,self.xset,self.yset)

#===============================================================================
# FUNCTION: s1d_output
#===============================================================================

def s1d_output(K_list,Te_list,K_name):
   """
   Create SymRateCoef output for the calculated rate coefficient. This form
   can be used by the ODE solver.
   """
   K_list = list(K_list)
   Te_list = list(Te_list)
   K_name = str(K_name)
   pT = sympy.Symbol("pT")
   spl_K = spline1D (Te_list,K_list,k=3)           
   K_src = SymRateCoeff(K_name,spl_K,spl_K.deriv)  
   K_src_plt = sympy.lambdify(pT,K_src(pT))
   return [spl_K,K_src,K_src_plt]

#===============================================================================
# CLASS:TheanoInterpWrapOp
#===============================================================================

class TheanoInterpWrapOp(theano.Op):
   def __init__(self,FuncName,SrcX,SrcY,Order=3):
      self.fname = FuncName
      self.spl_func = spline1D(SrcX,SrcY,k=Order)

   def __eq__(self, other):
      if type(self) == type(other):
         return self.fname == other.fname
      else:
         return False

   def __hash__(self,):
      return hash((type(self),self.fname))

   def __str__(self,):
      return self.__class__.__name__ + '.'+self.fname

   def make_node(self, x):
      x_ = theano.tensor.as_tensor_variable(x)
      assert x_.ndim <= 1
      return theano.Apply(self, inputs=[x_], outputs=[x_.type()])

   def perform(self, node, inputs, output_storage):
      x = inputs[0]
      z = output_storage[0]
      z[0] = self.spl_func(x)

   def infer_shape(self, node, i0_shapes):
      return i0_shapes


#===============================================================================
# Run test
#===============================================================================

if __name__ == "__main__":
  print ("Running test case ")
  k = TestInterpOp()
  k.CreateSuite()
  print '\n=======\nSuite Test 0\n======='
  f0 = k.CreateFunc(k.suite[0],k.suite[1],s=True,t=True,dimset=0)
  print '\n=======\nSuite Test 1\n======='
  f1 = k.CreateFunc(k.suite[0],k.suite[2],s=True,t=True,dimset=0)
  print '\n=======\nSuite Test 2\n======='
  f2 = k.CreateFunc(k.suite[0],k.suite[3],s=True,t=True,dimset=0)
  

