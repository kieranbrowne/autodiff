(ns autodiff.protocols)

(defprotocol AutoDiff
  (constant [u] "Create a constant of value u")
  (add [u v] "Add two values")
  (sub [u v] "Subtract two values")
  (mul [u v] "Multiply two values")
  (negate [u] "Invert sign of value; ie -2 -> 2; 9.3 -> - 9.3")
  (abs [u])
  (signum [u])
  (div [u v] "Divide one value by another")
  (recip [u])
  (pi [type-like] "Return constant value of pi")
  (one [type-like] "Return constant value equivalent to 1")
  (exp [u])
  (sqrt [u])
  (log [u])
  (sin [u])
  (cos [u])
  (tan [u])
  (asin [u])
  (acos [u])
  (atan [u])
  (sinh [u])
  (cosh [u])
  (tanh [u])
  (asinh [u])
  (acosh [u])
  (atanh [u])
  (pow [u v] "Raise one value to the power of another")
  (logbase [u v])
  )

(defrecord Dual
    [f f'])

(extend-type java.lang.Long
  AutoDiff
  (constant [a] a)
  )

(defn coerce
  "Makes value a Dual if not already"
  ([x v]
   (if (= (str (type x)) "class autodiff.protocols.Dual")
     x (->Dual x (constant v))))
  ([x] (coerce x 0)))

(defmacro destruct-unary
  "Simpler let bindings for autodiff of Dual record type"
  [content]
  `(let ~[{'u :f 'u' :f' :or {'u 'u 'u' 0}} 'u]
     ~content))

(defmacro destruct-binary
  "Simpler let bindings for autodiff of Dual record type"
  [content]
  `(let ~[{'u :f 'u' :f'} '(coerce u)
          {'v :f 'v' :f'} '(coerce v)]
     ~content))


(extend-type Dual
  AutoDiff
  (constant [u]
    (destruct-unary
     (Dual. (constant u) (constant 0))))
  (add [u v]
    (destruct-binary
      (Dual. (add u v) (add u' v'))))
  (sub [u v]
    (destruct-binary
      (Dual. (sub u v) (sub u' v'))))
  (mul [u v]
    (destruct-binary
      (Dual. (mul u v) (add (mul u' v) (mul u v')))))
  (div [u v]
    (destruct-binary
     (Dual. (div u v) (div (sub (mul u' v) (mul u v')) (mul v v)))))
  (sin [u]
    (destruct-unary
     (Dual. (sin u) (mul u' (cos u)))))
  (cos [u]
    (destruct-unary
     (Dual. (cos u) (negate (mul u' (sin u))))))
  (pi [type-like] (Dual. (pi type-like) 0))
  )


(defn d
  "Find the first derivative of a function"
  [f & args]
  (:f' (apply f (map #(coerce % 1) args))))
