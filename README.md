## Data-Wrangling-in-Python

### Common Data Types:

#### Numbers; Strings; Lists; Bytes objects and Bytearrays; Tuples; Sets; Dictionaries

### Statements: If Statement; While Statement; For Statement; Try Statement; With Statement; List Comprehension; Other Statements

### 1. Map Function
Python has an inbuilt function called map() which permits us to process all the elements present in an iterable without explicitly using a looping construct. When used, it returns a map object which in turn is an iterator. This map object is the result obtained by applying the specified function to every item present in the iterable.

Function definition – required_answer = map(function, iterable)

The map() function takes two arguments:

The first argument is a function that is to be applied to each and every element present in the iterable.
The second argument is the iterable itself on which the function is to be mapped.

###2. itertools
Python has an amazing standard library called itertools which provides a number of functions that help in writing clean, fast, and memory-efficient code due to lazy evaluation. It is a Python module that implements various iterator building blocks and together they form ‘iterator algebra’ which makes it possible to efficiently build tools in the Python language. The functions in itertools work on iterators itself which in turn return more complex iterators. Some example of functions present in itertools are: count(), cycle(), repeat(), accumulate(), product(), permutations(), combinations() etc. each taking their own set of arguments and operating upon them. The result is generated a lot more faster as compared to the results achieved when using conventional code.

### 3. Lambda Function
Python’s lambda functions are small anonymous functions as they do not have a name and are contained in a single line of code. The keyword ‘def’ is used to define functions in Python but lambda functions are rather defined by the keyword ‘lambda’. They can take any number of arguments, but the number of expressions can only be one. It makes code concise and easy to read for simple logical operations and is best to use when you need to use the function only a single time.

Function definition – required_answer = lambda ..arguments : expression

### 4. Exception Handling
Exceptions are types of errors that occur when the program is being executed and change the normal flow of the program. An example could be dividing a number by zero or referencing an index that is outside the bounds of an iterable. Therefore, we use try, except, and finally to handle exceptions in Python. The keyword try is used to wrap a block of code which can potentially throw errors, except is used to wrap a block of code to be executed when an exception is raised and handles the error and finally lets us execute the code no matter what. 

### 5. Decorators
Decorators are a part of Python’s metaprogramming which are used to add additional functionality to existing code without altering the original structure at compile time. It is more like a regular function in Python which can be called and returns a callable. It takes in a function, modifies it by adding functionality, and then returns it.

### 6. Collections
Collections in Python are general purpose inbuilt containers like set, tuple, dictionary, and lists. Python collections is a module that implements specialized container datatypes. Collections include namedtuple() which is a function for creating tuple subclasses with named fields, OrderedDict which is a dict subclass that remembers the order entries that were added since Python dict aren’t ordered, Counter that is used for counting hashable objects, ChainMap that is used for creating a single view of multiple mappings, etc. 

### 7. Generators
Generators in Python are a special type of function that rather than returning a single value, returns an iterator object which is a sequence of values. It is a utility to create your own iterator function. The keyword yield is used in the generator function instead of the return keyword which pauses its execution. The difference between yield and return is that return terminates the function but yield only pauses the execution of the function and returns the value against it each time. 

### 8. Magic Methods
Also called Dunder (or double underscore) methods, magic methods are special types of functions that are invoked internally. They start and end with double underscores. Some examples include __add__(), __abs__(), __round__(), __floor__(), __str__(), __trunc__(), __lshift__() etc. The expression number + 5 is the same as number.__add__(5) and this is internally called by other methods or actions. You can directly use these functions as it will decrease the run time of your code due to the fact that now due to direct use, we will be reducing a function call each time.

### 9. Threading
A Thread is the smallest unit or process that can be scheduled by an operating system. Python contains the Thread class which aids in multithreaded programming. Multithreading is mainly used to speed up computation up to a huge extent as now more than one thread will be performing tasks. To implement threading in Python, you will need to use the threading module (since the thread module is deprecated). 

### 10. Regular Expressions
Python regular expressions or RegEx are expressions that contain specific characters as patterns to be matched. It is used to check if a string or a set of strings contains a specific pattern. It is extremely powerful, elegant, and concise along with being fast. To use Python’s regular expressions, you need to import re module which contains functions that help in pattern matching like findall(), search(), split(), etc.

These were the top advanced Python concepts that you must know to be an experienced Python developer. These will not only make you a good programmer and developer but will also improve code readability and make it faster.
