# Auto Differentiation

[![Build Status](https://travis-ci.org/kieranbrowne/autodiff.svg?branch=master)](https://travis-ci.org/kieranbrowne/autodiff)

An attempt at auto diff.

## Usage

```clojure
(ns example.core
  (:refer-clojure :exclude  [* + - / identity])
  (:require [autodiff.core :refer :all]
            [autodiff.protocols :refer :all]
            ))
```

### Differentiation on simple expressions
```clojure
(+ 1 2 3)
; => 6
; derivative above expression with respect to 2
(d + 1 (wrt 2) 3)
; => 1
; derivative of 2*5 with respect to 2
(d * (wrt 2) 5)
; => 5
; derivative of 2*5 with respect to 2 & 5
(d * (wrt 5) (wrt 2))
; => 7
```

### Differentiation on functions of expressions
```clojure
(defn f [x] (* x 2))
; functions work as normal
(f 5)
; => 10
; but can be differentiated
(d f (wrt 5))
; => 2
; note that when applied to arguments without `wrt`
; values are assumed as constant
(d f 5)
; => 0
```

### Functions remain composable
```clojure
(defn f [x] (* x 3))
(defn g [x] (- x 1))
((comp f g) 5)
; => 12
(d (comp f g) (wrt 9))
; => 3
```


### Differentiation on complex expressions
```clojure
(reduce * 2 (map (partial * 2) [1 (wrt 2) 3]))
; => #Dual{:f 96 :f' 48}
; :f is the result of the function
; :f' is the derivative
```

## License

Copyright © 2017 Kieran Browne

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
